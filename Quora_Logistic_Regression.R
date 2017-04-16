library(stringdist)
setwd("/Users/Kalhan/Desktop/Waterloo Data/Winter 2017/MachineLearning/Quora/")
train_data <- read.csv('train-2.csv',stringsAsFactors = FALSE)
test_data <- read.csv('test-2.csv',stringsAsFactors = FALSE)


train_data <- as.data.frame(sapply(train_data , function(x) gsub("\"", "", x)))
train_data  <- as.data.frame(sapply(train_data , function(x) gsub("\'", " ", x)))
train_data <- as.data.frame(sapply(train_data, function(x) gsub(",", " ", x)))
train_data <- as.data.frame(sapply(train_data, function(x) gsub("\n", " ", x)))
train_data <- as.data.frame(sapply(train_data, function(x) gsub("</p>", " ", x)))
train_data  <- as.data.frame(sapply(train_data , function(x) gsub("<p>", " ", x)))
train_data  <- as.data.frame(sapply(train_data, function(x) gsub("\\?", " ", x)))

test_data <- as.data.frame(sapply(test_data , function(x) gsub("\"", "", x)))
test_data  <- as.data.frame(sapply(test_data , function(x) gsub("\'", " ", x)))
test_data <- as.data.frame(sapply(test_data, function(x) gsub(",", " ", x)))
test_data <- as.data.frame(sapply(test_data, function(x) gsub("\n", " ", x)))
test_data  <- as.data.frame(sapply(test_data, function(x) gsub("</p>", " ", x)))
test_data   <- as.data.frame(sapply(test_data , function(x) gsub("<p>", " ", x)))
test_data   <- as.data.frame(sapply(test_data, function(x) gsub("\\?", " ", x)))

cosine_sim <- lapply(train_data, function(x) stringsim(train_data$question1,train_data$question2, method = c("cosine")))


