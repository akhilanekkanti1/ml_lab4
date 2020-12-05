#copied from prelimfit2-ST
library(tidyverse)
library(tidymodels)
library(future)
library(rio)
library(vip)
library(here)
library(xgboost)



#import training data
train <- read_csv(here("data","train.csv"),
                  col_types = cols(.default = col_guess(), 
                                   calc_admn_cd = col_character()))  %>% 
  select(-classification) 

#import frl data - this is not the correct frl file
# frl <- import(here("data","frl.csv"),
#               setclass = "tbl_df")  %>% 
#   janitor::clean_names()  %>% 
#   filter(st == "OR")  %>%
#   #select(ncessch, lunch_program, student_count)  %>% # this is also the only other change from lab 4. I don't know if it would change anything though.
#   mutate(student_count = replace_na(student_count, 0))  %>% 
#   pivot_wider(names_from = lunch_program,
#               values_from = student_count)  %>% 
#   janitor::clean_names()  %>% 
#   mutate(ncessch = as.double(ncessch))

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


#create counts
stu_counts <- import(here("data","achievement-gaps-geocoded.csv"),
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))

#import ethnicity data
or_schools <- readxl::read_xlsx(here("data", "fallmembershipreport_20192020.xlsx"),
                                sheet = 4) 

#tidy ethnicity data
ethnicities <- or_schools %>% 
  select(attnd_schl_inst_id = `Attending School ID`,
         attnd_dist_inst_id = `Attending District Institution ID`, #included this to join by district along with school id
         sch_name = `School Name`,
         contains("%")) %>% 
  janitor::clean_names()
names(ethnicities) <- gsub("x2019_20_percent", "p", names(ethnicities))

#import staff data
staff <- import(here("data","staff.csv"),
                setclass = "tbl_df") %>% 
  janitor::clean_names() %>%
  filter(st == "OR") %>%
  select(ncessch,teachers) %>%
  mutate(ncessch = as.double(ncessch))




#joins
frl_stu <- left_join(frl, stu_counts)

frl_stu <- frl_stu %>% mutate(fl_prop = free_lunch_qualified/n,
                              rl_prop = reduced_price_lunch_qualified/n) %>%
  select(ncessch,fl_prop, rl_prop)

d <- train %>% 
  left_join(frl_stu) %>% 
  left_join(staff) %>% 
  left_join(ethnicities)

#TAKE OUT FOR TALAPAS
d <- d %>% 
  sample_frac(.01)

## Copied from lab5_talapas script
######################### split

d_split <- initial_split(d, strata = "score")

d_train <- training(d_split)
d_test  <- testing(d_split)
train_cv <- vfold_cv(d_train, strata = "score")

######################## recipe

rec_yoself1 <- recipe(score ~ .,data = d_train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt))) %>% #had  to add as.numeric to recipe to make it run
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

# rec_yoself %>% 
#   prep() %>% 
#   juice()

##########################TESTING XGBOOST MODEL############################ needed to edit recipe

cores <- parallel::detectCores()

xg_tunelr <- boost_tree() %>% 
  set_engine("xgboost", 
             nthreads = cores) %>% 
  set_mode("regression") %>% 
  set_args(trees = 5000,
           learn_rate = tune(), #tune learning rate first, then others
           stop_iter = 20)

xglr_flo <- workflow() %>%
  add_recipe(rec_yoself1) %>%
  add_model(xg_tunelr)

xg_grd <- expand.grid(learn_rate = seq(0.0001, 0.3, length.out = 20))

tictoc::tic()
tune_xglr <- tune_grid(xglr_flo, train_cv, grid = xg_grd, control = tune::control_resamples(verbose = TRUE, save_pred = TRUE))
tictoc::toc()

#2436.44 sec elapsed


#finalize lr with best RMSE and tune tree depth and min n

tune_xgtrn <- xg_tunelr %>% 
  finalize_model(select_best(tune_xglr)) %>% 
  set_args(tree_depth = tune(),
           min_n = tune())

xglr_flo <- xglr_flo %>% 
  update_model(tune_xgtrn)

trn_grd <- grid_max_entropy(tree_depth(), min_n(), size = 30)

tictoc::tic()
tune_xgtrn <- tune_grid(xglr_flo, train_cv, grid = trn_grd)
tictoc::toc()


#####COPIED FROM TALAPASPRELIM FIT - need to edit
##########################

# cores <- parallel::detectCores()
# 
# model_of_forests <- rand_forest() %>%
#   set_engine("ranger",
#              num.threads = cores, #argument from {ranger}
#              importance = "permutation", #argument from {ranger}
#              verbose = TRUE) %>% #argument from {ranger}
#   set_mode("regression") %>% 
#   set_args(mtry = tune(),
#            trees = 1000,
#            min_n = tune())
# 
# 
# forest_flo <- workflow() %>%
#   add_recipe(rec_yoself) %>%
#   add_model(model_of_forests)
# 
# ######################
# 
# set.seed(3000)
# plan(multisession)
# tictoc::tic()
# tune_random_trees <- tune_grid(forest_flo, 
#                                train_cv, 
#                                grid = 10,
#                                metrics = metric_set(rmse, rsq, huber_loss),
#                                control = control_resamples(verbose = TRUE, 
#                                                            save_pred = TRUE, 
#                                                            extract = function(x) x))
# 
# tictoc::toc()
# plan(sequential)
# 
# #######################
# 
# train_best <- select_best(tune_random_trees, metric = "rmse")
# 
# train_wf_final <- finalize_workflow(
#   forest_flo,
#   train_best
# )
# 
# tictoc::tic()
# set.seed(3000)
# train_res_final <- last_fit(train_wf_final,
#                             split = d_split)
# tictoc::toc()
# 
# train_res_final %>% 
#   collect_metrics()
# 
# ###########################
# test <- read_csv("data/test.csv", #edited path
#                  col_types = cols(.default = col_guess(), 
#                                   calc_admn_cd = col_character()))
# 
# 
# #joins - edited
# test1 <- test %>% 
#   left_join(frl_stu) %>% 
#   left_join(staff) %>% 
#   left_join(ethnicities)
# 
# #workflow
# fit_workflow <- fit(train_wf_final, d)
# 
# #use model to make predictions for test dataset
# preds_final <- predict(fit_workflow, test1)
# 
# ######################
# pred_frame <- tibble(Id = test1$id, Predicted = preds_final$.pred)
# 
# write_csv(pred_frame, "final-fit2.csv")
# 
# saveRDS(train_res_final, "prelimfit2_finalfit.Rds")













