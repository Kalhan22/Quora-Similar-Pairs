

##Logistic Regression
#Question Similarity
#model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col,family=binomial(link='logit'), data <- train_data_updated)
#Question and Noun/Verb Similarity
#model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n,family=binomial(link='logit'), data <- train_data_updated)
#Question and Noun/Verb Similarity Count(Question,Noun,Verb)
model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match,family=binomial(link='logit'), data <- train_data_updated)
logistic_model_prediction <- predict(model, test_data_updated, type = "prob") 
prediction_class <- ifelse(logistic_model_prediction > 0.5,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))


#Bootstrap
train_control <- trainControl(method="boot", number=100)
model <- train(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match ,data <- train_data_updated,trControl=train_control,method = "glm" )
print(model)


#Cross Validation
#Leave one out

train_control <- trainControl(method="LOOCV")
model <- train(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,family=binomial(link='logit'), trControl = train_control,data <- train_data_updated, method = "glm")
print(model)


#Repeated k fold
train_control <- trainControl(method="cv", number=10)
model <- train(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,family=binomial(link='logit'), trControl = train_control,data <- train_data_updated, method = "glm")
print(model)


#Repeated k fold
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,family=binomial(link='logit'), trControl = train_control,data <- train_data_updated, method = "glm")
print(model)



logistic_model_prediction <- predict(model, test_data_updated, type = "response") 
prediction_class <- ifelse(logistic_model_prediction > 0.5,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))

