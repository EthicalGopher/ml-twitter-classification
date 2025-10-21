# Twitter Classification Project

This project is part of the Codecademy Data Science career path. It explores two classification tasks using Twitter data:

1.  **Tweet Location Classification:** Predicting whether a tweet originated from New York, London, or Paris based on its text content.
2.  **Viral Tweet Prediction:** Predicting whether a tweet will go viral based on features like tweet length, follower count, and friend count.

## Project Structure

*   `tweet_location.ipynb`: A Jupyter notebook that implements a Naive Bayes classifier to predict the location of a tweet.
*   `viral_tweets.ipynb`: A Jupyter notebook that uses a K-Nearest Neighbors (KNN) classifier to predict if a tweet will go viral.
*   `london.json`, `new_york.json`, `paris.json`: JSON files containing tweets from London, New York, and Paris.
*   `random_tweets.json`: A JSON file containing a collection of random tweets.
*   `tweet_location_solution.ipynb`, `viral_tweets_solution.ipynb`: Solution notebooks provided by Codecademy.

## Tweet Location Classification

This part of the project uses a Naive Bayes classifier to determine the origin of a tweet (New York, London, or Paris).

### Approach

1.  **Data Loading:** The tweets from `new_york.json`, `london.json`, and `paris.json` are loaded into pandas DataFrames.
2.  **Feature Extraction:** The text of each tweet is used as the primary feature.
3.  **Model Training:** A `MultinomialNB` classifier from scikit-learn is trained on the tweet text. The text data is transformed into numerical data using `CountVectorizer`.
4.  **Evaluation:** The model's performance is evaluated using accuracy and a confusion matrix.

## Viral Tweet Prediction

This part of the project uses a K-Nearest Neighbors (KNN) classifier to predict whether a tweet will become "viral".

### Approach

1.  **Data Loading:** Tweets from `random_tweets.json` are loaded into a pandas DataFrame.
2.  **Label Creation:** A tweet is labeled as "viral" if its retweet count is above the median retweet count of all tweets in the dataset.
3.  **Feature Engineering:** The following features are created for each tweet:
    *   `tweet_length`: The number of characters in the tweet.
    *   `followers_count`: The number of followers of the user who posted the tweet.
    *   `friends_count`: The number of friends of the user who posted the tweet.
4.  **Data Normalization:** The features are normalized to ensure they are on a similar scale.
5.  **Model Training:** A `KNeighborsClassifier` from scikit-learn is trained on the engineered features.
6.  **Hyperparameter Tuning:** The model is tested with different values of `k` (the number of neighbors) to find the optimal value that yields the highest accuracy.

## How to Run

1.  Make sure you have Python 3, Jupyter Notebook, pandas, and scikit-learn installed.
2.  Open and run the cells in `tweet_location.ipynb` and `viral_tweets.ipynb` to see the analysis and results.
