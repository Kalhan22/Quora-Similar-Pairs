

#Read later
#https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine


##SVM Classification
#Question Similarity
#model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data <- train_data_updated)
#Question and Noun Verb Similarity
#model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n, data <- train_data_updated)
#Question,Noun,Verb Similarity, Count(Noun Verb Questio)
#Linear Kernel
model <- tune.svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,kernel = c("linear"),data <- train_data_updated)
svm_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- ifelse(svm_model_prediction > 0.6,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))



#Polynomial Kernel
model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,kernel = c("polynomial"),data <- train_data_updated)
svm_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- ifelse(svm_model_prediction > 0.6,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))


#Radial Kernel
model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,kernel = c("radial"),data <- train_data_updated)
svm_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- ifelse(svm_model_prediction > 0.6,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))


#Sigmoid Kernel
model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,kernel = c("sigmoid"),data <- train_data_updated)
svm_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- ifelse(svm_model_prediction > 0,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))


#Tuning SVM Polynomial
#Tuning Parameters
tuned_model <- tune(svm,train.x = train_data_updated[,11:ncol(train_data_updated)],train.y = train_data_updated[,6], ranges=list(cost=10^(-1:2), gamma = 10^(-6:-1)))
summary(tuned_model)
print(tuned_model)



#Running the best Parameters for Polynomial
#Polynomial Kernel
model <- svm(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,kernel = c("polynomial"), cost = 100, gamma = 0.00001,data <- train_data_updated)
svm_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- svm_model_prediction 
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))



