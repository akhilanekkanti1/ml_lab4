library(tidyverse)
library(tidymodels)
#library(doParallel)
library(rio)
library(glmnet)

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
  left_join(ethnicities) #%>% sample_frac(.01)

d <- train %>% 
  left_join(ethnicities) %>% 
  left_join(frl_stu) %>% 
  left_join(staff)

unqi_d <- d %>% 
  count(id) %>% 
  arrange(desc(n))

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


###MODEL

#linear regression model - tuned
lr_mod <- linear_reg()  %>%
  set_engine("glmnet") %>% 
  set_mode("regression") %>% 
  set_args(penalty = tune(), 
           mixture = tune())

#Workflow

lr_flo <- workflow() %>% 
  add_recipe(rec_yoself) %>% 
  add_model(lr_mod)

#######
set.seed(3000)

#grid
lr_grd <- grid_regular(penalty(), mixture(), levels = 30)


#tune model 

tictoc::tic()
lr_res <- tune::tune_grid(lr_flo, resamples = train_cv, grid = lr_grd,
                           control = tune::control_resamples(save_pred = TRUE))
tictoc::toc()


#select best tuning parameters
lr_best <- lr_res %>% 
  select_best(metric = "rmse")

lr_best

#finalize model in workflow
lr_flo_final <- finalize_workflow(lr_flo, lr_best)

#final fit

#evaluate on test set with last_fit
#This will automatically train the model specified by the workflow using the training data, and produce evaluations based on the test set.
lr_final_res <- last_fit(
  lr_flo_final,
  split = d_split)

lr_final_res %>%
  collect_metrics()

#get predictions from test set (within split object)
test_preds <- lr_final_res %>% collect_predictions()
test_preds


#copied from st-test
test <- read_csv("data/test.csv",
                 col_types = cols(.default = col_guess(), 
                                  calc_admn_cd = col_character())) %>% 
  select(-classification)

#joins - edited
test1 <- test %>% 
  left_join(frl_stu) %>% 
  left_join(staff) %>% 
  left_join(ethnicities)

unqi_test <- test1 %>% 
  count(id) %>% 
  arrange(n)

#If you want to use your model to predict the response for new observations, you need to use the fit() function on your workflow and the dataset that you want to fit the final model on (e.g. the complete training + testing dataset). 

#workflow
fit_workflow <- fit(lr_flo_final, d)#Should this be d_train based on lab 3key. This blog says not - http://www.rebeccabarter.com/blog/2020-03-25_machine_learning/

fit_workflow #view

#use model to make predictions for test dataset (adding test1 as new data)
preds_final <- predict(fit_workflow, test1)

######################
pred_frame <- tibble(Id = test1$id, Predicted = preds_final$.pred)

unqi_pred <- pred_frame %>% 
  count(Id) %>% 
  arrange(desc(n))

write_csv(pred_frame, "lrfit-m1-editrecipe.csv") #edited

saveRDS(lr_final_res, "lr_finalfit-m1.Rds")

#################################
## with code from lab 3 key - prepping and baking - i think workflow does this for you
# prepped_train <- rec_yoself %>% 
#   prep() %>% 
#   bake(d_train) %>% #this should be d_train according to notes which is different
#   select(-contains("id"), -ncessch) #added to get rid of NAs in predictions
# 
# prepped_test <- rec_yoself %>% 
#   prep() %>% 
#   bake(test1)
# 
# final_mod <- lr_mod %>% 
#   finalize_model(select_best(lr_res, metric = "rmse"))
# 
# full_train_fit <- fit(final_mod, score ~ ., prepped_train)
# 
# preds <- predict(full_train_fit, new_data = prepped_test)
# pred_file <- tibble(Id = test1$id, Predicted = preds$.pred) 
# write_csv(pred_file, "preds-enet.csv")

