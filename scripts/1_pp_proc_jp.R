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

set.seed(3000)
split <- initial_split(d)
train <- training(split)
train_cv <- vfold_cv(train)


# basic recipe
# decided on different variables, if we used stay_in_schl and ind_ed_fg there was little to no variance and was 
# deleted from the analyses

rec <- recipe(classification ~ econ_dsvntg + tag_ed_fg + enrl_grd + gndr + ethnic_cd, data = train)  %>% 
  step_mutate(gndr = as.factor(gndr),
              ethnic_cd = as.factor(ethnic_cd),
              enrl_grd = as.factor(enrl_grd),
              tag_ed_fg = as.factor(tag_ed_fg),
              econ_dsvntg = as.factor(econ_dsvntg),
              classification = ifelse(classification < 3, "below", "proficient")) %>% 
  step_unknown(all_nominal(), -all_outcomes())  %>% 
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>% 
  step_dummy(all_predictors(), -all_numeric(), -all_outcomes())  %>% 
  step_nzv(all_predictors())  


# linear regression model
lm <- nearest_neighbor()  %>%
  set_mode("classification") %>% 
  set_engine("kknn")
   

fit1jp <- fit_resamples(lm, rec, train_cv)
saveRDS(fit1jp, "fit1jp.Rds")