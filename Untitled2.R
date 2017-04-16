#Feature Selection Phase 1: Only Distance
#Take Sample
#Get Verbs, Nouns for Sample
#Get Other Features

options(java.parameters = "-Xmx16g" )
library(ipred)
library(openNLP)
library(caret)
library(dplyr)
library(NLP)
library(foreign)
library('e1071')
library('SparseM')
library('tm')
library(parallel)
library(stringdist)
library(party)
library(randomForest)
#library(RTextTools)
#library(caret)
#library(adabag)
# Calculate the number of cores
no_cores <- detectCores() - 1

# Initiate cluster
cl <- makeCluster(no_cores, type="FORK")

setwd("/Users/Kalhan/Desktop/Waterloo Data/Winter 2017/MachineLearning/Quora/")
train_data <- read.csv('train-2.csv',stringsAsFactors = FALSE)

all_duplicates <-  subset(train_data,train_data$is_duplicate == 1)
all_non_duplicates <- subset(train_data,train_data$is_duplicate == 0)
subset_duplicates <- all_duplicates[sample(nrow(all_duplicates), 370), ]
subset_non_duplicates <- all_non_duplicates[sample(nrow(all_non_duplicates), 630), ]

raw_data <- rbind(subset_duplicates,subset_non_duplicates)
raw_data <- raw_data[sample(nrow(raw_data)),]

final_test_set <- train_data[sample(nrow(train_data)), 1000]


tagPOS <-  function(x, ...) {
  s <- as.String(x)
  word_token_annotator <- Maxent_Word_Token_Annotator()
  a2 <- Annotation(1L, "sentence", 1L, nchar(s))
  a2 <- annotate(s, word_token_annotator, a2)
  a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
  a3w <- a3[a3$type == "word"]
  POStags <- unlist(parLapply(cl,a3w$features, `[[`, "POS"))
  POStagged <- paste(sprintf("%s/%s", s[a3w], POStags), collapse = " ")
  list(POStagged = POStagged, POStags = POStags)
}

getVerbs <- function(x,...){
  acqTag <- tagPOS(x)
  verb_q1 <-  paste(unique(sort(unlist(sapply(strsplit(as.character(acqTag),"[[:punct:]]*/VB.?"),function(x) {res = sub("(^.*\\s)(\\w+$)", "\\2", x); res[!grepl("\\s",res)]})))),collapse = " ")
}

getNouns <- function(x,...){
  acqTag <- tagPOS(x)
  verb_q1 <-  paste(unique(sort(unlist(sapply(strsplit(as.character(acqTag),"[[:punct:]]*/NN.?"),function(x) {res = sub("(^.*\\s)(\\w+$)", "\\2", x); res[!grepl("\\s",res)]})))),collapse = " ")
}


clusterEvalQ(cl, library(NLP))
clusterEvalQ(cl, library(openNLP))
clusterEvalQ(cl, library(foreign))
clusterEvalQ(cl, library('e1071'))
clusterEvalQ(cl, library('SparseM'))
clusterEvalQ(cl, library('tm'))
clusterEvalQ(cl, library(parallel))
clusterExport(cl, list("tagPOS", "getVerbs"))
clusterExport(cl, list("tagPOS", "getNouns"))
#clusterEvalQ(cl, "base")
#clusterExport(cl, list("tagPOS", "getVerbs"))

#sys <- proc.time();
#result_1 <- lapply(q2[1:100,],function(x) getVerbs(x))
#print(proc.time() - sys)

raw_data <- as.data.frame(sapply(raw_data , function(x) gsub("\"", "", x)))
raw_data  <- as.data.frame(sapply(raw_data , function(x) gsub("\'", " ", x)))
raw_data <- as.data.frame(sapply(raw_data, function(x) gsub(",", " ", x)))
raw_data <- as.data.frame(sapply(raw_data, function(x) gsub("\n", " ", x)))
raw_data <- as.data.frame(sapply(raw_data, function(x) gsub("</p>", " ", x)))
raw_data  <- as.data.frame(sapply(raw_data , function(x) gsub("<p>", " ", x)))
raw_data  <- as.data.frame(sapply(raw_data, function(x) gsub("\\?", " ", x)))


q2 <- raw_data$question2
q2<- as.character(q2)
q2 <- as.data.frame(q2)
colnames(q2) <- c("q2")


q1 <- raw_data$question1
q1<- as.character(q1)
q1 <- as.data.frame(q1)
colnames(q1) <- c("q1")

#Stemming
#q2 <- tolower(q2)
#q2 <- gsub("[?.;!¡¿·']", "", q2$q2)
#x <- wordStem(unlist(strsplit(q2,split = " ")))

result_verbs <- lapply(q2[1:nrow(q2),],function(x) getVerbs(x))
verbs <- as.data.frame(unlist(result_verbs))
colnames(verbs) <- c("all_verbs")
q2_verbs <- verbs
colnames(q2_verbs) <- c("all_verbs_q2")



result_noun <- lapply(q2[1:nrow(q2),],function(x) getNouns(x))
nouns <- as.data.frame(unlist(result_noun))
colnames(nouns) <- c("all_nouns")
q2_nouns <- nouns
colnames(q2_nouns) <- c("all_nouns_q2")



result_verbs <- lapply(q1[1:nrow(q1),],function(x) getVerbs(x))

#Loop Method
result_verbs <- c()
for(i in 1: nrow(q1)){
  temp <- getVerbs(q1[i,1])
  print(i)
  result_verbs <- c(result_verbs,temp)
}


verbs <- as.data.frame(result_verbs)
colnames(verbs) <- c("all_verbs")
q1_verbs <- verbs
colnames(q1_verbs) <- c("all_verbs_q1")


result_nouns <- lapply(q1[1:nrow(q1),],function(x) getNouns(x))
nouns <- as.data.frame(result_nouns)
colnames(nouns) <- c("all_nouns")
q1_nouns <- nouns
colnames(q1_nouns) <- c("all_nouns_q1")

#Loop Method
result_nouns <- c()
for(i in 1: nrow(q1)){
  temp <- getNouns(q1[i,1])
  print(temp)
  result_nouns <- c(result_nouns,temp)
}

#data_set_inter <- cbind(raw_data,q1_nouns,q2_nouns,q1_verbs,q2_verbs)
data_set_inter <- read.csv("data_1000.csv")

data_set_inter <- raw_data

#Similarity Question
cosine_sim <- lapply(data_set_inter,function(x) stringsim(data_set_inter$question1,data_set_inter$question2, method = c("cosine")))
jaccard_sim <- lapply(data_set_inter,function(x) stringsim(data_set_inter$question1,data_set_inter$question2, method = c("jaccard")))
jw_sim <- lapply(data_set_inter,function(x) stringsim(data_set_inter$question1,data_set_inter$question2, method = c("jw")))
soundex_sim <- lapply(data_set_inter,function(x) stringsim(data_set_inter$question1,data_set_inter$question2, method = c("soundex")))
qgram_sim <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$question1),as.character(data_set_inter$question2), method = c("qgram")))
lcs_sim <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$question1),as.character(data_set_inter$question2), method = c("lcs")))
hamming_sim <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$question1),as.character(data_set_inter$question2), method = c("hamming")))
dl_sim <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$question1),as.character(data_set_inter$question2), method = c("dl")))
lv_sim <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$question1),as.character(data_set_inter$question2), method = c("lv")))
osa_sim <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$question1),as.character(data_set_inter$question2), method = c("osa")))

#Similarity Nouns
cosine_sim_n <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_nouns_q1,data_set_inter$all_nouns_q2, method = c("cosine")))
jaccard_sim_n <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_nouns_q1,data_set_inter$all_nouns_q2, method = c("jaccard")))
jw_sim_n <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_nouns_q1,data_set_inter$all_nouns_q2, method = c("jw")))
soundex_sim_n <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_nouns_q1,data_set_inter$all_nouns_q2, method = c("soundex")))
qgram_sim_n <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_nouns_q1),as.character(data_set_inter$all_nouns_q2), method = c("qgram")))
lcs_sim_n <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_nouns_q1),as.character(data_set_inter$all_nouns_q2), method = c("lcs")))
hamming_sim_n <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_nouns_q1),as.character(data_set_inter$all_nouns_q2), method = c("hamming")))
dl_sim_n <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_nouns_q1),as.character(data_set_inter$all_nouns_q2), method = c("dl")))
lv_sim_n <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_nouns_q1),as.character(data_set_inter$all_nouns_q2), method = c("lv")))
osa_sim_n <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_nouns_q1),as.character(data_set_inter$all_nouns_q2), method = c("osa")))


#Similarity Verbs
cosine_sim_v <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_verbs_q1,data_set_inter$all_verbs_q2, method = c("cosine")))
jaccard_sim_v <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_verbs_q1,data_set_inter$all_verbs_q2, method = c("jaccard")))
jw_sim_v <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_verbs_q1,data_set_inter$all_verbs_q2, method = c("jw")))
soundex_sim_v  <- lapply(data_set_inter,function(x) stringsim(data_set_inter$all_verbs_q1,data_set_inter$all_verbs_q2, method = c("soundex")))
qgram_sim_v  <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_verbs_q1),as.character(data_set_inter$all_verbs_q2), method = c("qgram")))
lcs_sim_v <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_verbs_q1),as.character(data_set_inter$all_verbs_q2), method = c("lcs")))
hamming_sim_v <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_verbs_q1),as.character(data_set_inter$all_verbs_q2), method = c("hamming")))
dl_sim_v <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_verbs_q1),as.character(data_set_inter$all_verbs_q2), method = c("dl")))
lv_sim_v <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_verbs_q1),as.character(data_set_inter$all_verbs_q2), method = c("lv")))
osa_sim_v <- lapply(data_set_inter,function(x) stringsim(as.character(data_set_inter$all_verbs_q1),as.character(data_set_inter$all_verbs_q2), method = c("osa")))


#Question
cosine_sim_col <- as.data.frame(cosine_sim$id)
colnames(cosine_sim_col ) <- c("cosine_sim_col")
jaccard_sim_col <- as.data.frame(jaccard_sim$id)
colnames(jaccard_sim_col ) <- c("jaccard_sim_col")
jw_sim_col <- as.data.frame(jw_sim$id)
colnames(jw_sim_col ) <- c("jw_sim_col")
soundex_sim_col <- as.data.frame(soundex_sim$id)
colnames(soundex_sim_col ) <- c("soundex_sim_col")
qgram_sim_col <- as.data.frame(qgram_sim$id)
colnames(qgram_sim_col ) <- c("qgram_sim_col")
lcs_sim_col <- as.data.frame(lcs_sim$id)
colnames(lcs_sim_col ) <- c("lcs_sim_col")
hamming_sim_col <- as.data.frame(hamming_sim$id)
colnames(hamming_sim_col ) <- c("hamming_sim_col")
dl_sim_col <- as.data.frame(dl_sim$id)
colnames(dl_sim_col ) <- c("dl_sim_col")
lv_sim_col <- as.data.frame(lv_sim$id)
colnames(lv_sim_col ) <- c("lv_sim_col")
osa_sim_col <- as.data.frame(osa_sim$id)
colnames(osa_sim_col ) <- c("osa_sim_col")


#Verb
cosine_sim_col_v <- as.data.frame(cosine_sim_v$id)
colnames(cosine_sim_col_v ) <- c("cosine_sim_col_v")
jaccard_sim_col_v <- as.data.frame(jaccard_sim_v$id)
colnames(jaccard_sim_col_v ) <- c("jaccard_sim_col_v")
jw_sim_col_v <- as.data.frame(jw_sim_v$id)
colnames(jw_sim_col_v ) <- c("jw_sim_col_v")
soundex_sim_col_v <- as.data.frame(soundex_sim_v$id)
colnames(soundex_sim_col_v ) <- c("soundex_sim_col_v")
qgram_sim_col_v <- as.data.frame(qgram_sim_v$id)
colnames(qgram_sim_col_v ) <- c("qgram_sim_col_v")
lcs_sim_col_v <- as.data.frame(lcs_sim_v$id)
colnames(lcs_sim_col_v ) <- c("lcs_sim_col_v")
hamming_sim_col_v <- as.data.frame(hamming_sim_v$id)
colnames(hamming_sim_col_v ) <- c("hamming_sim_col_v")
dl_sim_col_v <- as.data.frame(dl_sim_v$id)
colnames(dl_sim_col_v ) <- c("dl_sim_col_v")
lv_sim_col_v <- as.data.frame(lv_sim_v$id)
colnames(lv_sim_col_v ) <- c("lv_sim_col_v")
osa_sim_col_v <- as.data.frame(osa_sim_v$id)
colnames(osa_sim_col_v ) <- c("osa_sim_col_v")


#Noun
cosine_sim_col_n <- as.data.frame(cosine_sim_n$id)
colnames(cosine_sim_col_n ) <- c("cosine_sim_col_n")
jaccard_sim_col_n <- as.data.frame(jaccard_sim_n$id)
colnames(jaccard_sim_col_n ) <- c("jaccard_sim_col_n")
jw_sim_col_n <- as.data.frame(jw_sim_n$id)
colnames(jw_sim_col_n ) <- c("jw_sim_col_n")
soundex_sim_col_n <- as.data.frame(soundex_sim_n$id)
colnames(soundex_sim_col_n ) <- c("soundex_sim_col_n")
qgram_sim_col_n <- as.data.frame(qgram_sim_n$id)
colnames(qgram_sim_col_n ) <- c("qgram_sim_col_n")
lcs_sim_col_n <- as.data.frame(lcs_sim_n$id)
colnames(lcs_sim_col_n ) <- c("lcs_sim_col_n")
hamming_sim_col_n <- as.data.frame(hamming_sim_n$id)
colnames(hamming_sim_col_n ) <- c("hamming_sim_col_n")
dl_sim_col_n <- as.data.frame(dl_sim_n$id)
colnames(dl_sim_col_n ) <- c("dl_sim_col_n")
lv_sim_col_n <- as.data.frame(lv_sim_n$id)
colnames(lv_sim_col_n ) <- c("lv_sim_col_n")
osa_sim_col_n <- as.data.frame(osa_sim_n$id)
colnames(osa_sim_col_n ) <- c("osa_sim_col_n")

#Number of Words in the Question

q2 <- raw_data$question2
q2<- as.character(q2)
q2 <- as.data.frame(q2)
colnames(q2) <- c("q2")


q1 <- raw_data$question1
q1<- as.character(q1)
q1 <- as.data.frame(q1)
colnames(q1) <- c("q1")

words_q1 <- lapply(as.character(q1$q1),function(x) length(unlist(strsplit(x, split = " "))))
words_q1 <- as.data.frame(unlist(words_q1))
colnames(words_q1) <- c("words_q1")

words_q2 <- lapply(as.character(q2$q2),function(x) length(unlist(strsplit(x, split = " "))))
words_q2 <- as.data.frame(unlist(words_q2))
colnames(words_q2) <- c("words_q2")



#Number Nouns
count_q2_nouns <- data_set_inter$all_nouns_q2
count_q2_nouns<- as.character(count_q2_nouns)
count_q2_nouns <- as.data.frame(count_q2_nouns)
colnames(count_q2_nouns) <- c("count_q2_nouns")


count_q1_nouns <- data_set_inter$all_nouns_q1
count_q1_nouns<- as.character(count_q1_nouns)
count_q1_nouns <- as.data.frame(count_q1_nouns)
colnames(count_q1_nouns) <- c("count_q1_nouns")

count_q1_nouns <- lapply(as.character(count_q1_nouns$count_q1_nouns),function(x) length(unlist(strsplit(x, split = " "))))
count_q1_nouns <- as.data.frame(unlist(count_q1_nouns))
colnames(count_q1_nouns) <- c("count_q1_nouns")

count_q2_nouns <- lapply(as.character(count_q2_nouns$count_q2_nouns),function(x) length(unlist(strsplit(x, split = " "))))
count_q2_nouns <- as.data.frame(unlist(count_q2_nouns))
colnames(count_q2_nouns) <- c("count_q2_nouns")


#Verbs
count_q2_verbs <- data_set_inter$all_verbs_q2
count_q2_verbs<- as.character(count_q2_verbs)
count_q2_verbs <- as.data.frame(count_q2_verbs)
colnames(count_q2_verbs) <- c("count_q2_verbs")


count_q1_verbs <- data_set_inter$all_verbs_q1
count_q1_verbs<- as.character(count_q1_verbs)
count_q1_verbs <- as.data.frame(count_q1_verbs)
colnames(count_q1_verbs) <- c("count_q1_verbs")

count_q1_verbs <- lapply(as.character(count_q1_verbs$count_q1_verbs),function(x) length(unlist(strsplit(x, split = " "))))
count_q1_verbs <- as.data.frame(unlist(count_q1_verbs))
colnames(count_q1_verbs) <- c("count_q1_verbs")

count_q2_verbs <- lapply(as.character(count_q2_verbs$count_q2_verbs),function(x) length(unlist(strsplit(x, split = " "))))
count_q2_verbs <- as.data.frame(unlist(count_q2_verbs))
colnames(count_q2_verbs) <- c("count_q2_verbs")



q1_first_string_data <- raw_data$question1
q1_first_string <-   lapply(q1_first_string_data,function(x) tolower(unlist(strsplit(as.character(x),split = " "))[1]))

q2_first_string_data <- raw_data$question2
q2_first_string <-   lapply(q2_first_string_data,function(x) tolower(unlist(strsplit(as.character(x),split = " "))[1]))


first_string_match <- data.frame(ifelse((unlist(q1_first_string) == unlist(q2_first_string)),1,0))
colnames(first_string_match) <- c("first_string_match")
first_string_match[is.na(first_string_match)] <- 0

raw_data_updated <- cbind(data_set_inter,cosine_sim_col,jaccard_sim_col,jw_sim_col,soundex_sim_col,qgram_sim_col,lcs_sim_col,hamming_sim_col,dl_sim_col,lv_sim_col,osa_sim_col,cosine_sim_col_n,jaccard_sim_col_n,jw_sim_col_n,soundex_sim_col_n,qgram_sim_col_n,lcs_sim_col_n,hamming_sim_col_n,dl_sim_col_n,lv_sim_col_n,osa_sim_col_n,cosine_sim_col_v,jaccard_sim_col_v,jw_sim_col_v,soundex_sim_col_v,qgram_sim_col_v,lcs_sim_col_v,hamming_sim_col_v,dl_sim_col_v,lv_sim_col_v,osa_sim_col_v,words_q1,words_q2,count_q1_verbs,count_q2_verbs,count_q1_nouns,count_q2_nouns,first_string_match)




train_data_updated <- raw_data_updated[1:800,]
test_data_updated <- raw_data_updated[801:1000,]


##Logistic Regression
#Question Similarity
#model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col,family=binomial(link='logit'), data <- train_data_updated)
#Question and Noun/Verb Similarity
#model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n,family=binomial(link='logit'), data <- train_data_updated)
#Question and Noun/Verb Similarity Count(Question,Noun,Verb)
model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match,family=binomial(link='logit'), data <- train_data_updated)
logistic_model_prediction <- predict(model, test_data_updated, type = "response") 
prediction_class <- ifelse(logistic_model_prediction > 0.5,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))

model <- glm(is_duplicate ~  jaccard_sim_col +  qgram_sim_col + lcs_sim_col   + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match,family=binomial(link='logit'), data <- train_data_updated)


#Logistic Regression
model <- glm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col  words_q1 + words_q2 +  first_string_match,family=binomial(link='logit'), data <- train_data_updated)
logistic_model_prediction <- predict(model, test_data_updated, type = "response") 
logistic_prediction_class <- ifelse(logistic_model_prediction > 0.5,1,0)
##SVM Classification
#Question Similarity
#model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data <- train_data_updated)
#Question and Noun Verb Similarity
#model <- svm(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n, data <- train_data_updated)
#Question,Noun,Verb Similarity, Count(Noun Verb Questio)
model <- tune.svm(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,kernel = "polynomial",cost = 100, gamma = 0.00001, data <- train_data_updated)
svm_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- svm_model_prediction
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))



##KNN Classification
library(class)

#Question and Noun Verb Similarity
#knn_model_prediction <- knn(train_data_updated[,11:40], test_data_updated[,11:40], cl = factor(train_data_updated$is_duplicate), k = 20)

#Question  Similarity
#knn_model_prediction <- knn(train_data_updated[,11:20], test_data_updated[,11:20], cl = factor(train_data_updated$is_duplicate), k = 20)
#Question Noun Verb Similarity and counts
#knn_model_prediction <- knn(train_data_updated[,11:ncol(train_data_updated)], test_data_updated[,11:ncol(test_data_updated)], cl = factor(train_data_updated$is_duplicate), k = 20)
prediction_class <- knn_model_prediction
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))



#Naive Bayes
#Question Similarity
model <- naiveBayes(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col,data <- train_data_updated)
#Question Noun/Verb Similarity
model <- naiveBayes(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n, data <- train_data_updated)

#Question,Noun,Verb Similarity, Count(Noun Verb Questio)
model <- naiveBayes(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match, data <- train_data_updated)
naive_model_prediction <- predict(model, test_data_updated,type = "class") 
prediction_class <- naive_model_prediction
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))



#Neural Net
library(nnet)
#Question Similarity
model <- nnet(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data <- train_data_updated,size=50,maxit=10000,decay=.002)

#Question Noun Verb Similarity
model <- nnet(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n,size=50,maxit=10000,decay=.002,MaxNWts = 10000, data <- train_data_updated)

#Question,Noun,Verb Similarity, Count(Noun Verb Questio)
model <- nnet(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns,size=50,maxit=10000,decay=.002,MaxNWts = 10000, data <- train_data_updated)


nnet_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- nnet_model_prediction
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))



#Decision Trees
library(rpart)
#Question Similarity
model <- rpart(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data <- train_data_updated)
#Question Noun Verb Similarity
model <- rpart(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n, data <- train_data_updated)

#Question,Noun,Verb Similarity, Count(Noun Verb Questio)
model <- rpart(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match, data <- train_data_updated)
dtree_model_prediction <- predict(model, test_data_updated, type = "prob") 
prediction_class <- ifelse(dtree_model_prediction > 0.5,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))


#RandomForest

#Question Similarity
model <- randomForest(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data <- train_data_updated)

#Question Noun Verb Similarity
model <- randomForest(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n, data <- train_data_updated)


#Question,Noun,Verb Similarity, Count(Noun Verb Questio)
model <- randomForest(as.factor(is_duplicate) ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns + first_string_match,mtry = 20, data <- train_data_updated)
forest_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- forest_model_prediction
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))


#Bagging
model <- bagging(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data = train_data_updated, nbag = 25)
model <- bagging(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n, data =  train_data_updated,nbag = 25)
model <- bagging(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n + words_q1 + words_q2 + count_q2_verbs + count_q1_verbs + count_q2_nouns + count_q1_nouns, data = train_data_updated,  nbag = 25)
bag_model_prediction <- predict(model, test_data_updated, type = "class") 
prediction_class <- ifelse(bag_model_prediction>0.5,1,0)
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))






#Boosting # 
#Question Similarity
library(adabag)
boost_train_data_updated <- train_data_updated 
boost_train_data_updated$is_duplicate <- as.factor(boost_train_data_updated$is_duplicate)
#model <- boosting(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col, data <- boost_train_data_updated , mfinal = 25,boos = TRUE, control = rpart.control(cp = -1))
#Queestion Noun Verb

model <- boosting(is_duplicate ~ cosine_sim_col + jaccard_sim_col + jw_sim_col + soundex_sim_col + qgram_sim_col + lcs_sim_col + hamming_sim_col + dl_sim_col + lv_sim_col + osa_sim_col + cosine_sim_col_v + jaccard_sim_col_v + jw_sim_col_v + soundex_sim_col_v + qgram_sim_col + lcs_sim_col_v + hamming_sim_col_v + dl_sim_col_v + lv_sim_col_v + osa_sim_col_v + cosine_sim_col_n + jaccard_sim_col_n + jw_sim_col_n + soundex_sim_col_n + qgram_sim_col_n + lcs_sim_col_n + hamming_sim_col_n + dl_sim_col_n + lv_sim_col_n + osa_sim_col_n,mfinal = 10, data <- boost_train_data_updated)
boost_model_prediction <- predict(model, test_data_updated) 
prediction_class <- boost_model_prediction$class
final_dataset <- cbind(test_data_updated,prediction_class)
print(100*sum(final_dataset$is_duplicate == final_dataset$prediction_class)/nrow(final_dataset))
print(confusionMatrix(as.factor(final_dataset$is_duplicate),as.factor( final_dataset$prediction_class)))




#Deep Nets in R
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
#library(mxnet)

train.x <- data.matrix(train_data_updated[,11:ncol(train_data_updated)])
train.y <- train_data_updated[,6]
test.x <- data.matrix(test_data_updated[,11:ncol(train_data_updated)])
test.y <- test_data_updated[,6]

mx.set.seed(9)
model <- mx.mlp(train.x, train.y, hidden_node=100, out_node=2, out_activation="softmax",
                num.round=20,learning.rate=0.07, momentum=0.9, 
                eval.metric=mx.metric.accuracy)

preds <-  predict(model, test.x)
pred.label <-  max.col(t(preds))-1
print(100*sum(final_dataset$is_duplicate == pred.label)/nrow(final_dataset))
table(pred.label, test.y)

stopCluster(cl)


