max <- 0
max_param <- 0
knn_graph <- data.frame(-1,-1)
colnames(knn_graph) <- c("k","accuracy")
for(param in 1:900){
  print(param)
  knn_model_prediction <- knn(train_data_updated[,15:ncol(train_data_updated)], test_data_updated[,15:ncol(test_data_updated)], cl = factor(train_data_updated$is_duplicate), k = param)
  prediction_class <- knn_model_prediction
  final_dataset <- cbind(test_data_updated,prediction_class)
  print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
  temp <- 100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset)
  knn_graph_row <- data.frame(param,temp)
  colnames(knn_graph_row) <- c("k","accuracy")
  knn_graph <- rbind(knn_graph,knn_graph_row)
  if(temp > max){
    max <- temp
    max_param <- param
    
  }
  #print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))
}
knn_graph <- subset(knn_graph,k!=-1)
plot(knn_graph)

#Max Accuracy Obtained 306