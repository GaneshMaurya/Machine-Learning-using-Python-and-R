#SVR 

# PRegression Template

#Data Preprocessing


#Importing Dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]


#Missing Data
# dataset$Age = ifelse(is.na(dataset$Age),
#                      ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Age)
# 
# dataset$Salary = ifelse(is.na(dataset$Salary),
#                      ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Salary)


#Encoding categorical data
# dataset$Country = factor(dataset$Country,
#                          levels = c("France", "Spain", "Germany"),
#                          labels = c(1, 2, 3))
# 
# dataset$Purchased = factor(dataset$Purchased,
#                            levels = c("No", "Yes"),


#Splitting the data into training set and test set
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)


#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


#Fitting Regression model
#install.packages("e1071")
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = "eps-regression")


#Predicting a new result with
y_pred = predict(regressor, data.frame(Level = 6.5))


#Visualizing Regression model results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = "blue") +
  ggtitle("Truth or Bluff (SVR Model)") + 
  xlab("Level of Experience") + 
  ylab("Salary")


#Visualizing Regression model results (high resolution curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = "blue") +
  ggtitle("Truth or Bluff (SVR Model)") + 
  xlab("Level of Experience") + 
  ylab("Salary") 

