models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")
library(caret)
library(dslabs)
library(tidyverse)

set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
data("mnist_27")

fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = mnist_27$train)
}) 

names(fits) <- models

predictions <- sapply(fits, function(fit){ 
  predict(fit, mnist_27$test)
}) 
predictions
nrow(predictions)
ncol(predictions)

accuracy <- colMeans(predictions == mnist_27$test$y)
mean(accuracy)

voting_voting <- rowSums(predictions == 7)/rowSums(predictions == 7 | predictions == 2)
voting_voting <- ifelse(voting >0.5, 7, 2)
voting_voting <- colMeans(predictions == mnist_27$test$y)
mean(voting_voting)

ind_accuracy <- sapply(models, function(model){
  mean(predictions[,model] == mnist_27$test$y)
})
ind_accuracy

cv_accuracy <- sapply(fits, function(fit) min(fit$results$Accuracy))
mean(cv_accuracy)


ind_accuracy <- sapply(models, function(model){
  mean(predictions[,model] == mnist_27$test$y)
})
ind_accuracy
