import pandas as pd
from sklearn import model_selection,feature_extraction,naive_bayes,metrics
df = pd.read_json("./twitter_classification_project/london.json",lines=True)
df1 = pd.read_json("./twitter_classification_project/new_york.json",lines=True)
df2 = pd.read_json("./twitter_classification_project/paris.json",lines=True)
df["label"] = 0
df1["label"] = 1
df2["label"] = 2
df = pd.concat([df,df1,df2])
all_tweets = df["text"].tolist()+df1["text"].tolist()+df2["text"].tolist()
label = df["label"].tolist()+df1["label"].tolist() + df2["label"].tolist()
df=df.drop(["display_text_range","in_reply_to_status_id","geo","quoted_status_id_str","in_reply_to_status_id_str","in_reply_to_screen_name","in_reply_to_user_id","in_reply_to_user_id_str","coordinates","contributors","extended_tweet","quoted_status_id","quoted_status_permalink","possibly_sensitive","quoted_status","withheld_in_countries","extended_entities"],axis=1)
train_data,test_data,train_label,test_label = model_selection.train_test_split(all_tweets,label,test_size = 0.2,random_state=1)
counter = feature_extraction.text.CountVectorizer()
train_counts = counter.fit_transform(train_data)
test_counts = counter.transform(test_data)
print(train_data[3])
print(train_counts[3])
classifier =  naive_bayes.MultinomialNB()
classifier.fit(train_counts,train_label)
print(classifier.predict(test_counts))
print(metrics.accuracy_score(classifier.predict(train_counts),train_label))
print(metrics.confusion_matrix(classifier.predict(train_counts),train_label))
