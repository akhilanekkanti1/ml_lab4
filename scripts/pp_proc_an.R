library(tidyverse)
library(tidymodels)
library(doParallel)

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})


d <- read_csv("train.csv") #%>% 
#  dplyr::sample_frac(.005)

set.seed(3000)
split <- initial_split(d)
train <- training(split)
train_cv <- vfold_cv(train)

# basic recipe
rec <- recipe(classification ~ stay_in_schl + tag_ed_fg + enrl_grd + ind_ed_fg, data = train)  %>% 
  step_mutate(enrl_grd = as.factor(enrl_grd),
              classification = as.factor(classification)) %>% 
  step_unknown(all_nominal(), -all_outcomes())  %>% 
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>% 
  step_dummy(all_predictors(), -all_numeric(), -all_outcomes())  %>% 
  step_nzv(all_predictors()) 

prep(rec)

# linear regression model
lm <- nearest_neighbor(neighbors = 11)  %>%
  set_engine("kknn") %>% 
  set_mode("classification")  

fit1 <- fit_resamples(lm, rec, train_cv)
saveRDS(fit1, "fit1.Rds")
