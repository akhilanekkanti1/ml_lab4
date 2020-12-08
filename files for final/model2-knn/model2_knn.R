###INITIAL SCRIPT

library(tidyverse)
library(tidymodels)
library(doParallel)
library(rio)

####### copied from st-test(random forest)
###################data

train <- read_csv("data/train.csv",
                  col_types = cols(.default = col_guess(), 
                                   calc_admn_cd = col_character()))  %>% 
  select(-classification) 


#edited (could export and then import that csv if needed)
frl <- import("https://nces.ed.gov/ccd/Data/zip/ccd_sch_033_1718_l_1a_083118.zip",
              setclass = "tbl_df")  %>% 
  janitor::clean_names()  %>% 
  filter(st == "OR")  %>%
  select(ncessch, lunch_program, student_count)  %>% 
  mutate(student_count = replace_na(student_count, 0))  %>% 
  pivot_wider(names_from = lunch_program,
              values_from = student_count)  %>% 
  janitor::clean_names()  %>% 
  mutate(ncessch = as.double(ncessch))

stu_counts <- import("data/achievement-gaps-geocoded.csv",
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))


or_schools <- readxl::read_xlsx("data/fallmembershipreport_20192020.xlsx",
                                sheet = 4) 

#tidy ethnicity data
ethnicities <- or_schools %>% 
  select(attnd_schl_inst_id = `Attending School ID`,
         attnd_dist_inst_id = `Attending District Institution ID`, #included this to join by district along with school id
         sch_name = `School Name`,
         contains("%")) %>% 
  janitor::clean_names()
names(ethnicities) <- gsub("x2019_20_percent", "p", names(ethnicities))

staff <- import("data/staff.csv",
                setclass = "tbl_df") %>% 
  janitor::clean_names() %>%
  filter(st == "OR") %>%
  select(ncessch,schid,teachers) %>%
  mutate(ncessch = as.double(ncessch))



frl_stu <- left_join(frl, stu_counts)

frl_stu <- frl_stu %>% mutate(fl_prop = free_lunch_qualified/n,
                              rl_prop = reduced_price_lunch_qualified/n) %>%
  select(ncessch,fl_prop, rl_prop)


d <- train %>% 
  left_join(frl_stu) %>% 
  left_join(staff) %>% 
  left_join(ethnicities)

######################### split

d_split <- initial_split(d, strata = "score")

d_train <- training(d_split)
d_test  <- testing(d_split)
train_cv <- vfold_cv(d_train, strata = "score")

######################## recipe

rec_yoself <- recipe(score ~ .,data = d_train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt))) %>% #had  to add as.numeric to recipe to make xgboost model run
  update_role(contains("id"), ncessch, new_role = "id vars") %>%
  step_unknown(all_nominal()) %>% 
  step_novel(all_nominal()) %>% 
  step_dummy(all_nominal()) %>% 
  step_nzv(all_predictors()) %>%
  #step_mutate(z_rlprop = log(rl_prop),
  #           z_flprop = log(fl_prop)) %>% 
  #step_rm(fl_prop, rl_prop) %>% #remove potentially redundant variables
  step_normalize(all_numeric(), -all_outcomes(), -has_role("id vars")) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars")) %>%  
  step_interact(terms = ~lat:lon) %>% 
  step_nzv(all_predictors()) #added due to error in xg boost about constant variables with 0sd

######

# basic recipe - old recipe
# 
# rec <- recipe(classification ~ econ_dsvntg + tag_ed_fg + enrl_grd + gndr + ethnic_cd, data = train)  %>% 
#   step_mutate(gndr = as.factor(gndr),
#               ethnic_cd = as.factor(ethnic_cd),
#               enrl_grd = as.factor(enrl_grd),
#               tag_ed_fg = as.factor(tag_ed_fg),
#               econ_dsvntg = as.factor(econ_dsvntg),
#               classification = ifelse(classification < 3, "below", "proficient")) %>% 
#   step_unknown(all_nominal(), -all_outcomes())  %>% 
#   step_novel(all_nominal(), -all_outcomes()) %>%
#   step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>% 
#   step_dummy(all_predictors(), -all_numeric(), -all_outcomes())  %>% 
#   step_nzv(all_predictors())  

###MODEL

#knn model- slide 80
knn_mod <- nearest_neighbor()  %>%
  set_engine("kknn") %>% 
  set_mode("regression") %>% 
  set_args(neighbors = tune())#,
           #weight_func = tune()#,
           #dist_power = tune())

translate(knn_mod)
#Workflow

knn_flo <- workflow() %>% 
  add_recipe(rec_yoself) %>% 
  add_model(knn_mod)

#######
set.seed(3000)

#grid
#knn_par <- parameters(neighbors(range = (c(10, 75))), weight_func(), dist_power()) #testing with smaller range due to computation
knn_grd <- grid_max_entropy(neighbors(), size = 30) #testing with smaller size due to computation


all_cores <- parallel::detectCores(logical = FALSE)

#tune model 
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

tictoc::tic()
knn_res <- tune::tune_grid(knn_flo, resamples = train_cv, grid = knn_grd,
                           control = tune::control_resamples(save_pred = TRUE))
parallel::stopCluster(cl)
tictoc::toc()


#select best tuning parameters
knn_best <- knn_res %>% 
  select_best(metric = "rmse")

#finalize model in workflow
knn_flo_final <- finalize_workflow(knn_flo, knn_best)

#final fit

registerDoSEQ() #need to unregister parallel processing in order to use all_nominal()
# cl <- makeCluster(8)
# registerDoParallel(cl)
knn_final_res <- last_fit(
  knn_flo_final,
  split = d_split)
# stopCluster(cl)

knn_final_res %>%
  collect_metrics()

#copied from st-test
test <- read_csv("data/test.csv",
                 col_types = cols(.default = col_guess(), 
                                  calc_admn_cd = col_character()))

#joins - edited
test1 <- test %>% 
  left_join(frl_stu) %>% 
  left_join(staff) %>% 
  left_join(ethnicities)

#workflow
fit_workflow <- fit(knn_flo_final, d)

#use model to make predictions for test dataset
preds_final <- predict(fit_workflow, test1)

######################
pred_frame <- tibble(Id = test1$id, Predicted = preds_final$.pred)

write_csv(pred_frame, "knnfit-m2-editrecipe.csv") #edited

saveRDS(knn_final_res, "knn_finalfit-m2.Rds")

# ##FINAL FIT- NEED TO UPDATE
# 
# knn_best1 <- readRDS("knn2_fit.Rds")
# 
# knn_best <- knn_best1 %>% 
#   select_best(metric = "roc_auc") 
# 
# # Finalize your model using the best tuning parameters
# knn_mod_final <- knn2_mod %>%
#   finalize_model(knn_best)
# 
# # Finalize your recipe using the best turning parameters
# knn_rec_final <- rec %>%
#   finalize_recipe(knn_best)
# 
# #run final fit
# 
# registerDoSEQ() #need to unregister parallel processing in order to use all_nominal()
# knn_final_res <- last_fit(
#   knn_mod_final,
#   preprocessor = knn_rec_final,
#   split = split)

#ASJDKLASJDKLA


