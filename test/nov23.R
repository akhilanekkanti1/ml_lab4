library(tidyverse)
library(rio)
library(here)
library(tidymodels)
library(doParallel)
library(workflows)

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

full_train <- read_csv(here("data","train.csv"),
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

d <- left_join(full_train, frl) %>%
  sample_frac(.001)

set.seed(3000)
(d_split <- initial_split(d)) 

d_train <- training(d_split)
d_test  <- testing(d_split)

set.seed(3000)
d_cv <- vfold_cv(d_train, v = 10)

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
             num.threads = 28, #argument from {ranger}
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

ourgrid <- grid_regular(mtry(range = c(1, 10)),
                        min_n(range = c(4, 40)),
                        levels = c(5, 5)) 

set.seed(3000)
##
workflow <- werkflo %>%
  update_model(you_call_that_a_model)


fit_recyoself_again <- tune_grid(
  workflow,
  resamples = d_cv,
  grid = ourgrid,
  metrics = metric_set(rmse),
  control = control_resamples(verbose = TRUE, #prints model fitting process
                              save_pred = TRUE, #saves out of sample predictions
                              extract = function(x) x)) 

# mean = 89.82, se = .58, mtry = 4, min_n = 40 .1 sample frac

oob <- fit_recyoself_again %>%
  mutate(oob = map_dbl(.extracts,
                       ~pluck(.x$.extracts, 1)$fit$fit$fit$prediction.error)) %>%
  select(id, oob)

fit_recyoself_again %>%  collect_metrics(summarize = FALSE)

best <- select_best(fit_recyoself_again, metric = "rmse")

finalwf <- finalize_workflow(
  werkflo,
  best
)

set.seed(3000)
finalbest <- last_fit(finalwf,
                      split = d_split)

predictions <- pluck(finalbest$.predictions[[1]])
write.csv(train_predictions, "predictions.csv")

full_test <- read_csv("../input/edld-654-fall-2020/test.csv",
                      col_types = cols(.default = col_guess(), 
                                       calc_admn_cd = col_character()))


##################test predictions
testdata <- read_csv(here("data", "test.csv"), 
                     col_types = cols(.default = col_guess(), 
                      calc_admn_cd = col_character()))

frl_test <- left_join(testdata, frl)

fit_workflow <- fit(workflow, frl_test)
