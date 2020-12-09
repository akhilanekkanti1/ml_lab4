
getwd()
setwd("C:/Users/strevino/Desktop/ML Labs-GitHub/ml_lab4")

lr_rds <- readRDS("model1-lr.Rds")

knn_rds <- readRDS("model2-knn.Rds")

rf_rds <- readRDS("model3-rf.Rds")



#Extract spec model args

knn_rds$.workflow
#10 neighbors
#1772.517 sec elapsed - from talapas

# rf_rds$.workflow
# 
# options(max.print=1000000000)

str(rf_rds$.workflow)
#mtry = 5, min_n = 40, trees = 1000
#13919.058 sec elapsed - from talapas

#linear reg results - from talapas .Rout
# Penalty = 0.0000000001  Mixture = 0.0345
#1552.898 sec elapsed