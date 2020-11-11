library(tidyverse)
library(tidymodels)
library(doParallel)

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

set.seed(3000)
d <- read_csv(here::here("data", "train.csv")) %>% 
  dplyr::sample_frac(.15)

set.seed(3000)
split <- initial_split(d)
train <- training(split)
train_cv <- vfold_cv(train)

# basic recipe
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


knn2 <- nearest_neighbor()  %>%
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           dist_power = tune())

knn_params <- parameters(neighbors(range = c(1,20)), dist_power())
knn_gridmax <- grid_max_entropy(knn_params, size = 25)

knn2_fit <- tune::tune_grid(
  knn2,
  preprocessor = rec,
  resamples = train_cv,
  grid = knn_gridmax,
  control = tune::control_resamples(save_pred = TRUE)
)

######################FINALFIT

knn_best <- knn2_fit %>% 
  select_best(metric = "roc_auc") 

knn_mod_final <- knn2 %>%
  finalize_model(knn_best)

knn_rec_final <- rec %>%
  finalize_recipe(knn_best)

registerDoSEQ() 
knn_final_res <- last_fit(
  knn_mod_final,
  preprocessor = knn_rec_final,
  split = split)

saveRDS(knn2_final_res, "knn2_finalfit_an.Rds")


