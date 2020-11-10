library(tidyverse)
library(tidymodels)
library(doParallel)

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})


d <- read_csv("train.csv") 
  dplyr::sample_frac(.005)

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


#slide64-66, 70-72 - create parameters and grid

knn_params <- parameters(neighbors(range = c(1,20)), weight_func(), dist_power())
knn_gridmax <- grid_max_entropy(knn_params, size = 25)

#ggplot of knn_gridmax

knn_gridmax %>% 
  ggplot(aes(neighbors, dist_power)) +
           geom_point() 
ggsave("neighbors_dist.png", path = here::here("plots"))




                      