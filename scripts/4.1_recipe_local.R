library(tidyverse)
library(rio)
library(tidymodels)
library(here)
library(doParallel)

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

full_train <- read_csv("data/train.csv",
                       col_types = cols(.default = col_guess(), 
                        calc_admn_cd = col_character()))  %>% 
              select(-classification)


frl <- import(here("data","frl.csv"),
              setclass = "tbl_df")  %>% 
  janitor::clean_names()  %>% 
  filter(st == "OR")  %>%
  select(ncessch, lunch_program, student_count)  %>% 
  mutate(student_count = replace_na(student_count, 0))  %>% 
  pivot_wider(names_from = lunch_program,
              values_from = student_count)  %>% 
  janitor::clean_names()  %>% 
  mutate(ncessch = as.double(ncessch))

stu_counts <- import(here("data","achievement-gaps-geocoded.csv"),
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))

staff <- import(here("data","staff.csv"),
                setclass = "tbl_df") %>% 
  janitor::clean_names() %>%
  filter(st == "OR") %>%
  select(ncessch,schid,teachers) %>%
  mutate(ncessch = as.double(ncessch))

frl <- left_join(frl, stu_counts)
frl1 <- left_join(frl, staff)

frl <- frl1 %>% 
  mutate(fl_prop = free_lunch_qualified/n,
         rl_prop = reduced_price_lunch_qualified/n) %>%
  select(ncessch,fl_prop, rl_prop, teachers)

d <- left_join(full_train, frl) 

set.seed(3000)
(d_split <- initial_split(d)) 

d_train <- training(d_split)
d_test  <- testing(d_split)

set.seed(3000)

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
  step_normalize(rl_prop, fl_prop, teachers) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars")) %>% #medianimpute only proportion variables 
  step_interact(terms = ~lat:lon)

halfbaked <- rec_yoself %>%
  prep %>%
  bake(d_train)

halfbaked

set.seed(3000)

(cores <- parallel::detectCores())

model_of_forests <- rand_forest() %>%
  set_engine("ranger",
             num.threads = cores, #argument from {ranger}
             importance = "permutation", #argument from {ranger}
             verbose = TRUE) %>% #argument from {ranger}
  set_mode("regression")

werkflo <- workflow() %>%
  add_recipe(rec_yoself) %>%
  add_model(model_of_forests)

set.seed(3000)
fit_recyoself <- fit_resamples(
  werkflo,
  resamples = d_cv,
  metrics = metric_set(rmse),
  control = control_resamples(verbose = TRUE, #prints model fitting process
                              save_pred = TRUE, #saves out of sample predictions
                              extract = function(x) x))

fit_recyoself %>%
  collect_metrics()
#oldrec: mean = 88.73802 , n = 10 , std_error = 0.2426
# mean = 90.55, se = .58, sample frac = .1 LM
# mean = 90.85, se = .66, sample frac .1 rando forest


you_call_that_a_model <- model_of_forests %>%
  set_args(mtry = tune(),
           trees = 1000,
           min_n = tune())

workflow <- werkflo %>%
  update_model(you_call_that_a_model)


fit_recyoself_again <- tune_grid(
  workflow,
  resamples = d_cv,
  grid = 10,
  metrics = metric_set(rmse),
  control = control_resamples(verbose = TRUE, #prints model fitting process
                              save_pred = TRUE, #saves out of sample predictions
                              extract = function(x) x)) 

fit_recyoself_again %>%
  collect_metrics() %>% 
  write_csv("final_rmse.csv")
# mean = 89.82, se = .58, mtry = 4, min_n = 40 .1 sample frac

rmse <- fit_recyoself_again %>%
  collect_metrics()
write_csv(rmse, "final_rmse.csv")

########
full_fit <- fit(you_call_that_a_model, score ~ ., data = select(halfbaked, -contains("id"), -ncessch))
  
  #read in test data
test <- read_csv("data/test.csv",
                   col_types = cols(.default = col_guess(), 
                                    calc_admn_cd = col_character()))
  
  #join with frl
test <- left_join(test, frl, by = "ncessch")
  
#bake with test set
halfbaked_test <- rec_yoself %>%
    prep() %>%
    bake(test)

#fit lm_mod from above using freshbaked (baked training set)
final_fit <- fit(you_call_that_a_model, score ~ ., data = select(halfbaked, -contains("id"), -ncessch))
#predictions with baked test set
predictions <- predict(final_fit, new_data = halfbaked_test)
  
#write out predictions to submit
  
final_predictions <- tibble(Id = halfbaked_test$id, Predicted = predictions$.pred)
write_csv(final_predictions, "final_predictions.csv")