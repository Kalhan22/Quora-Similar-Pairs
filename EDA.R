#Zeroes and Ones by Levels
x <- data.frame(raw_data_updated$cosine_sim_col,raw_data_updated$is_duplicate)
colnames(x) <- c("cosine_sim","is_duplicate")
library(sqldf)
ones <- sqldf("select Level, count(is_duplicate) as ones from (select 
                                            case when cosine_sim > 0.9 then 10
                                                 when cosine_sim > 0.8 then 9 
                                                 when cosine_sim > 0.7 then 8
                                                 when cosine_sim > 0.6 then 7 
                                                 when cosine_sim > 0.5 then 6 
                                                 when cosine_sim > 0.4 then 5 
                                                 when cosine_sim > 0.3 then 4 
                                                 when cosine_sim > 0.2 then 3
                                                 when cosine_sim > 0.1 then 2
                                                 else 1
                                            end as Level,is_duplicate
                                        from  x
                                              )
                as a  where is_duplicate = 1 group by 1

                ")

zeroes <- sqldf("select Level, count(is_duplicate) as zeroes from (select 
                                            case when cosine_sim > 0.9 then 10
              when cosine_sim > 0.8 then 9 
              when cosine_sim > 0.7 then 8
              when cosine_sim > 0.6 then 7 
              when cosine_sim > 0.5 then 6 
              when cosine_sim > 0.4 then 5 
              when cosine_sim > 0.3 then 4 
              when cosine_sim > 0.2 then 3
              when cosine_sim > 0.1 then 2
              else 1
              end as Level,is_duplicate
              from  x
)
as a  where is_duplicate = 0 group by 1

")

together <- sqldf("select a.Level, a.ones, b.zeroes from ones as a join zeroes as b on (a.Level = b.Level)")


#First String
#q1_first_string_data <- raw_data$question1
q1_first_string <-   data.frame(unlist(lapply(q1_first_string_data,function(x) tolower(unlist(strsplit(as.character(x),split = " "))[1]))))
colnames(q1_first_string) <- c("q1_first_string")

q2_first_string_data <- raw_data$question2
q2_first_string <-   data.frame(unlist(lapply(q2_first_string_data,function(x) tolower(unlist(strsplit(as.character(x),split = " "))[1]))))
colnames(q2_first_string) <- c("q2_first_string")

question_1_type <- sqldf("select q1_first_string,count(*) as occurences from q1_first_string group by 1 order by 2 desc ")

question_2_type <- sqldf("select q2_first_string,count(*) as occurences from q2_first_string group by 1 order by 2 desc ")

overall_question_type <- sqldf("select a.q1_first_string,a.occurences as q1_occurences,b.occurences as q2_occurences from question_1_type as a join question_2_type as b on (a.q1_first_string = b.q2_first_string) ")

write.csv(overall_question_type,"overall_question_type.csv")


#Questions Type with Duplicate and Non Duplicate Scenario
dummy_data <- data.frame(q1_first_string$q1_first_string,q2_first_string$q2_first_string,raw_data_updated$is_duplicate)
colnames(dummy_data) <- c("question_type1","question_type2","is_duplicate")

result <- sqldf("
                select 'How' as question_type,is_duplicate,count(*) as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'how' group by 1,2
                union
                select 'What' as question_type,is_duplicate,count(*) as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'what' group by 1 ,2
                union
                select 'Why' as question_type,is_duplicate,count(*) as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'why' group by 1,2 
                union
                select 'Is' as question_type,is_duplicate,count(*) as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'is' group by 1,2
                union
                select 'Which' as question_type,is_duplicate,count(*)  as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'which' group by 1 ,2
                union
                select 'Can' as question_type,is_duplicate,count(*)  as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'can' group by 1 ,2
                union
                select 'Do' as question_type,is_duplicate,count(*) as occurences from dummy_data where question_type1 == question_type2 and question_type1== 'do' group by 1,2
                ")

#Average Similar Nouns/Verbs/Strings/Punctuation
train_data <- read.csv('train-2.csv',stringsAsFactors = FALSE)

all_duplicates <-  subset(train_data,train_data$is_duplicate == 1)
all_non_duplicates <- subset(train_data,train_data$is_duplicate == 0)
subset_duplicates <- all_duplicates[sample(nrow(all_duplicates), 370), ]
subset_non_duplicates <- all_non_duplicates[sample(nrow(all_non_duplicates), 630), ]


pure_raw_data <- rbind(subset_duplicates,subset_non_duplicates)
pure_raw_data <- pure_raw_data[sample(nrow(pure_raw_data)),]

raw_q2 <- pure_raw_data$question2
raw_q2<- as.character(raw_q2)
raw_q2 <- as.data.frame(raw_q2)
colnames(raw_q2) <- c("raw_q2")


raw_q1 <-  pure_raw_data$question1
raw_q1<- as.character(q1)
raw_q1 <- as.data.frame(q1)
colnames(raw_q1) <- c("raw_q1")


punc_length_q1 <- data.frame(unlist(lapply(as.character(raw_q1$raw_q1),function(x) abs(nchar(as.vector(x),type = "bytes") - nchar(gsub("[[:punct:]]", "", as.vector(x)),type = "bytes")  ))))
colnames(punc_length_q1) <- c("punc_length_q1")

punc_length_q2 <- data.frame(unlist(lapply(as.character(raw_q2$raw_q2),function(x) nchar(as.vector(x),type = "bytes") - nchar(gsub("[[:punct:]]", "", as.vector(x)),type = "bytes"))))
colnames(punc_length_q2) <- c("punc_length_q2")

result <- data.frame(punc_length_q1$punc_length_q1,punc_length_q2$punc_length_q2,raw_data_updated$is_duplicate)
colnames(result) <- c("punc_q1","punc_q2","is_duplicate")

result_updated <- data.frame(data.frame(abs(result$punc_q2 - result$punc_q1)), result$is_duplicate)
colnames(result_updated) <- c("Punctuation_Difference","Duplicate")

average_diff <- sqldf("select Duplicate,AVG(Punctuation_Difference) as Average_PD from result_updated group by 1")




result_updated_verbs <- data.frame(data.frame(abs(raw_data_updated$count_q2_verbs - raw_data_updated$count_q1_verbs)), raw_data_updated$is_duplicate)
colnames(result_updated_verbs) <- c("Verb_Difference","Duplicate")
average_diff_verbs <- sqldf("select Duplicate,AVG(Verb_Difference) as Average_VD from result_updated_verbs group by 1")




result_updated_nouns <- data.frame(data.frame(abs(raw_data_updated$count_q1_nouns - raw_data_updated$count_q2_nouns)), raw_data_updated$is_duplicate)
colnames(result_updated_nouns) <- c("Noun_Difference","Duplicate")
average_diff_nouns <- sqldf("select Duplicate,AVG(Noun_Difference) as Average_ND from result_updated_nouns group by 1")



result_updated_words <- data.frame(data.frame(abs(raw_data_updated$words_q2 - raw_data_updated$words_q1)), raw_data_updated$is_duplicate)
colnames(result_updated_words) <- c("Word_Difference","Duplicate")
average_diff_words <- sqldf("select Duplicate,AVG(Word_Difference) as Average_WD from result_updated_words group by 1")


#Word Cloud Formation of Similar Words

train_data <- read.csv('train-2.csv',stringsAsFactors = FALSE,fileEncoding = "UTF-8")

all_duplicates <-  subset(train_data,train_data$is_duplicate == 1)
all_non_duplicates <- subset(train_data,train_data$is_duplicate == 0)
subset_duplicates <- all_duplicates[sample(nrow(all_duplicates), 370), ]
subset_non_duplicates <- all_non_duplicates[sample(nrow(all_non_duplicates), 630), ]

wrong_raw_data <- rbind(subset_duplicates,subset_non_duplicates)
wrong_raw_data <- wrong_raw_data[sample(nrow(wrong_raw_data)),]



corpus_similar <- sqldf("select * from wrong_raw_data where is_duplicate = 1")
corpus_different <- sqldf("select * from wrong_raw_data where is_duplicate = 0")
topics_similar <- c(as.character(corpus_similar$question1),as.character(corpus_similar$question2))
topics_different <- c(as.character(corpus_different$question1),as.character(corpus_different$question2))

topics_similar <- as.data.frame(topics_similar)
topics_different <- as.data.frame(topics_different)



jeopCorpus_1 <- Corpus(VectorSource(topics_similar$topics_similar))
jeopCorpus_2 <- Corpus(VectorSource(topics_different$topics_different))

jeopCorpus_1 <- tm_map(jeopCorpus_1, content_transformer(tolower))
jeopCorpus_2 <- tm_map(jeopCorpus_2, content_transformer(tolower))

jeopCorpus_1 <- tm_map(jeopCorpus_1, removePunctuation)
jeopCorpus_1 <- tm_map(jeopCorpus_1, PlainTextDocument)
jeopCorpus_1 <- tm_map(jeopCorpus_1, removeWords, stopwords('english'))

jeopCorpus_2 <- tm_map(jeopCorpus_2, removePunctuation)
jeopCorpus_2 <- tm_map(jeopCorpus_2, PlainTextDocument)
jeopCorpus_2 <- tm_map(jeopCorpus_2, removeWords, stopwords('english'))

jeopCorpus_1 <- tm_map(jeopCorpus_1, stemDocument)
jeopCorpus_2 <- tm_map(jeopCorpus_2, stemDocument)

jeopCorpus_1 <- tm_map(jeopCorpus_1, removeWords, c("what","how","which","where","can","will","whose","is","do","can","best"))
jeopCorpus_2 <- tm_map(jeopCorpus_2, removeWords, c("what","how","which","where","can","will","whose","is","do","can","best"))


library(wordcloud)

#Similar Question Pair Topics
wordcloud(jeopCorpus_1, max.words = 30,min.freq=5,scale=c(4,.5), 
          random.order = FALSE,rot.per=.5,vfont=c("sans serif","plain"),colors=palette()) 

#Dis similar Question Pair Topics
wordcloud(jeopCorpus_2, max.words = 30,min.freq=5,scale=c(4,.5), 
          random.order = FALSE,rot.per=.5,vfont=c("sans serif","plain"),colors=palette()) 