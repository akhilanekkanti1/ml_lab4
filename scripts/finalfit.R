
knn_better <- readRDS("knn2_fit_local.Rds")

knn_best <- knn2_fit %>%
  select_best(metric = "roc_auc") 

# Finalize your model using the best tuning parameters
knn_mod_final <- knn2_mod %>%
  finalize_model(knn_best)

# Finalize your recipe using the best turning parameters
knn_rec_final <- knn2_fit %>%
  finalize_recipe(knn_best)
