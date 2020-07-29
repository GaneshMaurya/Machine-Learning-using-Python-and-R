# Polynomial Regression 

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


#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


#Fitting the Linear Regression model
lin_reg = lm(formula = Salary ~ Level,
               data = dataset)


#Fitting Polynomial Regression model
dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4
# dataset$Level5 = dataset$Level ^ 5
poly_reg = lm(formula = Salary ~ .,
              data = dataset)


#Visualizing Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = "blue") +
  ggtitle("Truth or Bluff (Linear Regression)") + 
  xlab("Level of Experience") + 
  ylab("Salary")


#Visualizing Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = "blue") +
  ggtitle("Truth or Bluff (Polynomial Regression)") + 
  xlab("Level of Experience") + 
  ylab("Salary")


#Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))


#Predicting a new result with Polynomial Regression
y_pred_2 = predict(poly_reg, data.frame(Level = 6.5,
                                        Level2 = 6.5^2,
                                        Level3 = 6.5^3,
                                        Level4 = 6.5^4))

