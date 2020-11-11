knn_best1 <- readRDS("knn2_fit.Rds")

knn_best <- knn_best1 %>% 
  select_best(metric = "roc_auc") 

# Finalize your model using the best tuning parameters
knn_mod_final <- knn2 %>%
  finalize_model(knn_best)

# Finalize your recipe using the best turning parameters
knn_rec_final <- rec %>%
  finalize_recipe(knn_best)

#run final fit

registerDoSEQ() #need to unregister parallel processing in order to use all_nominal()
knn_final_res <- last_fit(
  knn_mod_final,
  preprocessor = knn_rec_final,
  split = split)

saveRDS(knn_final_res, "knn2_finalfit.Rds")

