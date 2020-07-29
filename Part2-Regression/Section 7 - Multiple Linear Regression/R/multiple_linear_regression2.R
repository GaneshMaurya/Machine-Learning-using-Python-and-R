#Multiple Linear Regression

#Data Preprocessing


#Importing Dataset
dataset = read.csv("50_Startups.csv")
# dataset = dataset[, 2:3]


#Missing Data
# dataset$Age = ifelse(is.na(dataset$Age),
#                      ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Age)
# 
# dataset$Salary = ifelse(is.na(dataset$Salary),
#                      ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Salary)


#Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1, 2, 3))



#Splitting the data into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


#Fitting Multiple Linear Regression on the training set
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
#regressor = lm(formula = Profit ~ .,
#               data = training_set)


#Predicting Test set resluts
y_pred = predict(regressor, newdata = test_set)


#Building optimal model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)
