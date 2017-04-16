
max <- 0
max_param <- 0
nnet_graph <- data.frame(-1,-1)
colnames(nnet_graph) <- c("HL_Nodes","accuracy")
for(param in 1:100){
  model <- nnet(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match,size=param,maxit=10000,decay=.002,MaxNWts = 10000, linout = FALSE ,Hess = TRUE,data <- train_data_updated)
  #Hessian = True fives the best result
  nnet_model_prediction <- predict(model, test_data_updated, type = "class") 
  prediction_class <- nnet_model_prediction
  final_dataset <- cbind(test_data_updated,prediction_class)
  temp <- 100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset)
  nnet_graph_row <- data.frame(param,temp)
  colnames(nnet_graph_row) <- c("HL_Nodes","accuracy")
  nnet_graph <- rbind(nnet_graph,nnet_graph_row)
  if(temp > max){
    max <- temp
    max_param <- param
    
  }
  #print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
  #print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))
}
nnet_graph <- subset(nnet_graph,HL_Nodes!=-1)
plot(nnet_graph)
write.csv(nnet_graph,"nnet_graph.csv")

nnet_graph <- read.csv("nnet_graph.csv")
