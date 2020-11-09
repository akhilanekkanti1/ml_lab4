library(tidyverse)
library(tidymodels)
library(doParallel)

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})


d <- read_csv("data/train.csv") %>% 
  dplyr::sample_frac(.005)

split <- initial_split(d)
train <- training(split)
train_cv <- vfold_cv(train)

# basic recipe
rec <- recipe(classification ~ stay_in_schl + tag_ed_fg + enrl_grd + ind_ed_fg, data = train)  %>% 
  step_mutate(enrl_grd = as.factor(enrl_grd)) %>% 
  step_unknown(all_nominal(), -all_outcomes())  %>% 
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>% 
  step_dummy(all_predictors(), -all_numeric(), -all_outcomes())  %>% 
  step_nzv(all_predictors()) 

# linear regression model
lm <- nearest_neighbor()  %>% 
  set_mode("classification")  %>% 
  set_engine("kknn")

fit1 <- fit_resamples(lm, rec, train_cv)
saveRDS(fit1, "fit1.Rds")
