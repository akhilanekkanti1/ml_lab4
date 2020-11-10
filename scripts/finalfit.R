
knn_best <- knn2_fit %>%
  select_best(metric = "roc_auc") 

# Finalize your model using the best tuning parameters
knn_mod_final <- knn2 %>%
  finalize_model(knn_best)

# Finalize your recipe using the best turning parameters
knn_rec_final <- rec %>%
  finalize_recipe(knn_best)

#run final fit
cl <- makeCluster(8)
registerDoParallel(cl)

knn_final_res <- last_fit(
  knn_mod_final,
  preprocessor = knn_rec_final,
  split = split)

stopCluster(cl)

#Collect metrics
knn_final_res %>%
  collect_()
