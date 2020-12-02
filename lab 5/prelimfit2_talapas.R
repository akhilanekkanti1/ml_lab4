library(tidyverse)
library(tidymodels)
library(baguette)
library(future)
library(rio)
library(vip)
library(ranger)


###################data

train <- read_csv("train.csv",
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

stu_counts <- import("achievement-gaps-geocoded.csv",
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))


or_schools <- readxl::read_xlsx("fallmembershipreport_20192020.xlsx",
                                sheet = 4) 

ethnicities <- or_schools %>% 
  select(attnd_schl_inst_id = `Attending School ID`,
         sch_name = `School Name`,
         contains("%")) %>% 
  janitor::clean_names()
names(ethnicities) <- gsub("x2019_20_percent", "p", names(ethnicities))

staff <- import("staff.csv",
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
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt)) %>%
  update_role(contains("id"), ncessch, new_role = "id vars") %>%
  step_unknown(all_nominal()) %>% 
  step_novel(all_nominal()) %>% 
  step_dummy(all_nominal()) %>% 
  step_nzv(all_predictors()) %>%
  #step_mutate(z_rlprop = log(rl_prop),
  #           z_flprop = log(fl_prop)) %>% 
  #step_rm(fl_prop, rl_prop) %>% #remove potentially redundant variables
  step_normalize(all_numeric(), -all_outcomes(), -has_role("id vars")) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars")) %>% #medianimpute only proportion variables 
  step_interact(terms = ~lat:lon)

rec_yoself %>% 
  prep() %>% 
  juice()

##########################

cores <- parallel::detectCores() ###not sure if we need to take this out and hard code num.threads below for talapas

model_of_forests <- rand_forest() %>%
  set_engine("ranger",
             num.threads = cores, #argument from {ranger}
             importance = "permutation", #argument from {ranger}
             verbose = TRUE) %>% #argument from {ranger}
  set_mode("regression") %>% 
  set_args(mtry = tune(),
           trees = 1000,
           min_n = tune())


forest_flo <- workflow() %>%
  add_recipe(rec_yoself) %>%
  add_model(model_of_forests)

######################

set.seed(3000)
plan(multisession)
tictoc::tic()
tune_random_trees <- tune_grid(forest_flo, 
                               train_cv, 
                               grid = 10,
                               metrics = metric_set(rmse, rsq, huber_loss),
                               control = control_resamples(verbose = TRUE, 
                                                           save_pred = TRUE, 
                                                           extract = function(x) x))

tictoc::toc()
plan(sequential)

#######################

train_best <- select_best(tune_random_trees, metric = "rmse")

train_wf_final <- finalize_workflow(
  forest_flo,
  train_best
)

tictoc::tic()
set.seed(3000)
train_res_final <- last_fit(train_wf_final,
                            split = d_split)
tictoc::toc()

train_res_final %>% 
  collect_metrics()

###########################
test <- read_csv("test.csv",
                 col_types = cols(.default = col_guess(), 
                                  calc_admn_cd = col_character()))

#joins - edited
test1 <- test %>% 
  left_join(frl_stu) %>% 
  left_join(staff) %>% 
  left_join(ethnicities)

#workflow
fit_workflow <- fit(train_wf_final, d)

#use model to make predictions for test dataset
preds_final <- predict(fit_workflow, test1)

######################
pred_frame <- tibble(Id = test1$id, Predicted = preds_final$.pred)

write_csv(pred_frame, "fit2-final.csv") #edited

saveRDS(train_res_final, "prelimfit2_finalfit.Rds")#added to save RDS object for final




