import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import neighbors

df = pd.read_json("./twitter_classification_project/random_tweets.json",lines=True)
df["isViral"]=np.where(df["retweet_count"]>5,1,0)

df["tweet_length"] = df.apply(lambda tweet: len(tweet['text']),axis=1)
df["followers_count"] =df.apply(lambda followers:followers['user']['followers_count'],axis=1)
df["hashtags_count"] = df.apply(lambda tweet:tweet["text"].count("#"),axis=1)
df["links_count"] = df.apply(lambda tweet:tweet["text"].count("http"),axis=1)
df["words_count"] = df.apply(lambda tweet:len(tweet["text"].split(" ")),axis=1)

data = df[["tweet_length","followers_count","hashtags_count","links_count","words_count"]]
scaled_data = scale(data,axis=0)
train_data,test_data,train_label,test_label = train_test_split(scaled_data,df["isViral"],test_size=0.2,random_state=1)
classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_data,train_label)
print(classifier.score(test_data,test_label))
