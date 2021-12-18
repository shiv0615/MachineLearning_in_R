library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

#Partition the data with 80-20 split
set.seed(42, sample.kind = "Rounding") # if using R 3.6 or later
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2, list = FALSE) # create a 20% test set
test_set <- titanic_clean[test_index,]
train_set <- titanic_clean[-test_index,]

# Get intuition for the data -> what would a random sample predict
set.seed(3)
survived <- sample(c(0,1),length(test_index),replace=TRUE)%>%
  factor(levels = levels(test_set$Survived))
mean(survived==test_set$Survived)

#Derive intuition for important variables
female_survivied <- train_set %>% filter(Sex=='female' & Survived == 1) %>% select('Survived')
male_survivied <- train_set %>% filter(Sex=='male' & Survived == 1) %>% select('Survived')
count(female_survivied)/count(train_set$Survived==1)
count(male_survivied)/sum(train_set$Survived==1)

survival_proportion_train_set_by_sex <- train_set %>% group_by(Sex) %>% 
  summarize(average=mean(Survived==1))
survival_proportion_train_set_by_sex

survival_proportion_test_set_sex <- test_set %>% group_by(Sex) %>% 
  summarize(average=mean(Survived==1))
survival_proportion_test_set_sex

y_hat_sex <- ifelse(test_set$Sex == 'female',1,0) %>%
  factor(levels = levels(test_set$Survived))
y_hat_sex <- factor(y_hat_sex)
mean(y_hat_sex == test_set$Survived)
confusionMatrix(y_hat_sex,test_set$Survived)
F_meas(data = y_hat_sex, reference = test_set$Survived)

survival_proportion_train_set_by_class <- train_set %>% group_by(Pclass) %>% 
  summarize(average=mean(Survived==1))
survival_proportion_train_set_by_class

y_hat_pclass <- ifelse(test_set$Pclass == 1,1,0) %>%
  factor(levels = levels(test_set$Survived))
y_hat_pclass <- factor(y_hat_pclass)
mean(y_hat_pclass == test_set$Survived)
confusionMatrix(y_hat_pclass,test_set$Survived)
F_meas(data = y_hat_pclass, reference = test_set$Survived)

survival_proportion_train_set_by_sexclass <- train_set %>% group_by(Pclass, Sex) %>% 
  summarize(average=mean(Survived==1))
survival_proportion_train_set_by_sexclass

indx_Pclass <- (test_set$Pclass == 1 | test_set$Pclass == 2)
indx_female <- (test_set$Sex == 'female')
indx        <- (indx_female & indx_Pclass)
y_hat_sex_pclass <- ifelse(indx, 1, 0) %>%
  factor(levels = levels(test_set$Survived))
y_hat_sex_pclass <- factor(y_hat_sex_pclass)
mean(y_hat_sex_pclass == test_set$Survived)
confusionMatrix(y_hat_sex_pclass,test_set$Survived)
F_meas(data = y_hat_sex_pclass, reference = test_set$Survived)

#Try Different Training Models
set.seed(1)
train_lda_fare <- train(Survived ~ Fare, method = "lda", data = train_set)
y_hat_lda_fare <- predict(train_lda_fare, test_set)
confusionMatrix(y_hat_lda_fare, test_set$Survived)$overall["Accuracy"]

set.seed(1)
train_qda_fare <- train(Survived ~ Fare, method = "qda", data = train_set)
y_hat_qda_fare <- predict(train_qda_fare, test_set)
confusionMatrix(y_hat_qda_fare, test_set$Survived)$overall["Accuracy"]

set.seed(1)
train_glm_age <- train(Survived ~ Age, method = "glm", data = train_set)
y_hat_glm_age <- predict(train_glm, test_set)
confusionMatrix(y_hat_glm_age, test_set$Survived)$overall["Accuracy"]

set.seed(1)
train_glm_many <- train(Survived ~ Sex+Pclass+Fare+Age, method = "glm", data = train_set)
y_hat_glm_many <- predict(train_glm_many, test_set)
confusionMatrix(y_hat_glm_many, test_set$Survived)$overall["Accuracy"]

set.seed(1)
train_glm_all <- train(Survived ~ ., method = "glm", data = train_set)
y_hat_glm_all <- predict(train_glm_all, test_set)
confusionMatrix(y_hat_glm_all, test_set$Survived)$overall["Accuracy"]

#KNN Model
set.seed(6)
train_knn_cv_all <- train(Survived ~ ., method = "knn", data = train_set, tuneGrid = data.frame(k = seq(3, 51, 2)),)
ggplot(train_knn_cv_all, highlight = TRUE)
train_knn_cv_all$bestTune
y_hat_knn_cv_all <- predict(train_knn_cv_all, test_set)
confusionMatrix(y_hat_knn_cv_all, test_set$Survived)$overall["Accuracy"]

#KNN Model
set.seed(8)
fitControl <- trainControl(method = "cv", number = 10)
train_knn_cv_all <- train(Survived ~ ., method = "knn", 
                          data = train_set, 
                          tuneGrid = data.frame(k = seq(3, 51, 2)),
                          trControl=fitControl)
ggplot(train_knn_cv_all, highlight = TRUE)
train_knn_cv_all
train_knn_cv_all$bestTune
y_hat_knn_cv_all <- predict(train_knn_cv_all, test_set)
confusionMatrix(y_hat_knn_cv_all, test_set$Survived)$overall["Accuracy"]

#Tree Base Model
set.seed(10)
fitControl <- trainControl(method = "cv", number = 10)
train_tree_cv_all <- train(Survived ~ ., method = "rpart", 
                          data = train_set, 
                          tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))
ggplot(train_tree_cv_all, highlight = TRUE)
train_tree_cv_all
train_tree_cv_all$bestTune
plot(train_tree_cv_all$finalModel, margin = 0.1)
text(train_tree_cv_all$finalModel, cex = 0.75)
y_hat_tree_cv_all <- predict(train_tree_cv_all, test_set)
confusionMatrix(y_hat_tree_cv_all, test_set$Survived)$overall["Accuracy"]

#RF Model
set.seed(10)
train_rf_cv_all <- train(Survived ~ ., method = "rf", 
                           data = train_set, 
                           tuneGrid = data.frame(mtry = seq(1:7)),
                           ntree=100)
ggplot(train_rf_cv_all, highlight = TRUE)
train_rf_cv_all
train_rf_cv_all$bestTune
plot(train_rf_cv_all$finalModel, margin = 0.1)
text(train_rf_cv_all$finalModel, cex = 0.75)
y_hat_rf_cv_all <- predict(train_rf_cv_all, test_set)
confusionMatrix(y_hat_rf_cv_all, test_set$Survived)$overall["Accuracy"]
varImp(train_rf_cv_all)
