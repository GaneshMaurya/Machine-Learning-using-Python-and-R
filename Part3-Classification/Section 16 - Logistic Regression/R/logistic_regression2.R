#Logistic Regression

#Importing Dataset
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[, 3:5]


#Splitting the data into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


#Feature Scaling
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])


#Fitting the Logistic Regression to the training set
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)


#Predicting Test Set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred>0.5, 1, 0)


#Making the confusion matrix
cm = table(test_set[, 3], y_pred)


#Visualizing the Training set results


















