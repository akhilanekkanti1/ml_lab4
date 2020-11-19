library(tidyverse)
library(rio)
library(tidymodels)
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


frl <- read_csv("data/frl.csv",
              setclass = "tbl_df")  %>% 
  janitor::clean_names()  %>% 
  filter(st == "OR")  %>%
  select(ncessch, lunch_program, student_count)  %>% 
  mutate(student_count = replace_na(student_count, 0))  %>% 
  pivot_wider(names_from = lunch_program,
              values_from = student_count)  %>% 
  janitor::clean_names()  %>% 
  mutate(ncessch = as.double(ncessch))

stu_counts <- read_csv("data/achievement-gaps-geocoded.csv",
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))

staff <- read_csv("data/staff.csv",
                setclass = "tbl_df") %>% 
  janitor::clean_names() %>%
  filter(st == "OR") %>%
  select(ncessch,schid,teachers) %>%
  mutate(ncessch = as.double(ncessch))

frl <- left_join(frl, stu_counts)
frl1 <- left_join(frl, staff)


set.seed(3000)
split <- initial_split(d)
train <- training(split)
train_cv <- vfold_cv(train)

#####
frl <- frl1 %>% 
  mutate(fl_prop = free_lunch_qualified/n,
         rl_prop = reduced_price_lunch_qualified/n) %>%
  select(ncessch,fl_prop, rl_prop, teachers)

d <- left_join(full_train, frl)

set.seed(3000)
(d_split <- initial_split(d)) 

d_train <- training(d_split)
d_test  <- testing(d_split)

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
  step_interact(terms = ~lat:lon) %>%
  step_ns(fl_prop, rl_prop,teachers, deg_free = 20)

prep(rec_yoself)

halfbaked <- rec_yoself %>%
  prep %>%
  bake(d_train) %>%
  print(n = 10)

(cores <- parallel::detectCores())

mod <-
  rand_forest() %>%
  set_engine("ranger",
             num.threads = cores, #argument from {ranger}
             importance = "permutation", #argument from {ranger}
             verbose = TRUE) %>% #argument from {ranger}
  set_mode("regression")

set.seed(3000)
#sub_dtrain <- d_train %>% sample_frac(.01)
d_cv <- vfold_cv(d_train, v = 10)

fit_recyoself <- fit_resamples(
  object = mod,
  preprocessor = rec_yoself,
  resamples = d_cv,
  control = control_resamples(verbose = TRUE, #prints model fitting process
                              save_pred = TRUE)) #saves out of sample predictions

fit_recyoself %>%
  collect_metrics()
#oldrec: mean =88.73802 , n = 10 , std_error = 0.2426 (prelim fit)
#newrec: mean = 96.69 (15 splines) - subset
#newrec: mean = 97.10 (20 splines) - subset

#tuned mod
tune_mod <-
  rand_forest() %>%
  set_engine("ranger",
             num.threads = cores, #argument from {ranger}
             importance = "permutation", #argument from {ranger}
             verbose = TRUE) %>% #argument from {ranger}
  set_mode("regression") %>%
  set_args(mtry = tune(),
           trees = 1000, #can tune 
           min_n = tune())


#copied grid
rf_tune_res <- tune_grid(
  tune_mod,
  tune_rec_yoself,
  d_cv,
  grid = 20,
  control = control_resamples(verbose = TRUE,
                              save_pred = TRUE))
                              #extract = function(x) extract_model(x)) #need workflow for extract?

  
  
  #tuned recipe - copied
  tune_rec_yoself <- recipe(score ~ .,data = d_train) %>%
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
    step_interact(terms = ~lat:lon) %>%
    step_ns(fl_prop, rl_prop,teachers, deg_free = tune())


