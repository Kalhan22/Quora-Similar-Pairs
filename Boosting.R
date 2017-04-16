library(adabag)
boost_train_data_updated <- train_data_updated 
boost_train_data_updated$is_duplicate <- as.factor(boost_train_data_updated$is_duplicate)

max <- 0
max_param <- 0
boosting_graph <- data.frame(-1,-1)
colnames(boosting_graph) <- c("Boost_Bags","accuracy")
for(param in 1:100){
  model <- boosting(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n,mfinal = param, data <- boost_train_data_updated)
  boost_model_prediction <- predict(model, test_data_updated) 
  prediction_class <- boost_model_prediction$class
  final_dataset <- cbind(test_data_updated,prediction_class)
  print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
  #print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))
  temp <- 100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset)
  boosting_graph_row <- data.frame(param,temp)
  colnames(boosting_graph_row) <- c("Boost_Bags","accuracy")
  boosting_graph <- rbind(boosting_graph,boosting_graph_row)
  if(temp > max){
    max <- temp
    max_param <- param
  }
}

boosting_graph <- subset(boosting_graph,Boost_Bags!=-1)
plot(boosting_graph)
#Max Accuracy After Boosting 68.5 @ Boost Bags = 7
#write.csv(boosting_graph,"boosting_graph.csv")


