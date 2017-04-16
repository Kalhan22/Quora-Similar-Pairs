#Multilayer Perceptron
#http://tjo-en.hatenablog.com/entry/2016/03/30/233848



cols <- c(1:ncol(train.x))

# Standardize
library(plyr)
standardize <- function(x) as.numeric((x - mean(x)) / sd(x))



train.x <- train_data_updated[,15:ncol(train_data_updated)]
train.x[cols] <- plyr::colwise(standardize)(train.x[cols])
train.x <- data.matrix(train.x)
train.y <- train_data_updated[,6]


test.x <- test_data_updated[,15:ncol(train_data_updated)]
test.x[cols] <- plyr::colwise(standardize)(test.x[cols])
test.x <- data.matrix(test.x)
test.y <- test_data_updated[,6]





data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
devices <- mx.cpu()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,ctx=devices, num.round=10, array.batch.size=100,learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
    initializer=mx.init.uniform(0.07), epoch.end.callback=mx.callback.log.train.metric(100))
         
  
preds <-  predict(model, test.x)
pred.label <-  max.col(t(preds))-1
print(100*sum(final_dataset$is_duplicate == pred.label)/nrow(final_dataset))
print(confusionMatrix(as.factor(test_data_updated$is_duplicate),as.factor(pred.label)))

