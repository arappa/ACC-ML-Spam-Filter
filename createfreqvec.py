import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load vocabulary
vocabulary = {}
with open("vocabulary.txt", "r") as vocab_file:
    for line in vocab_file:
        word, index = line.strip().split("\t")
        vocabulary[word] = int(index)

# Load the dataset
data_sample = pd.read_csv("data\emails.csv")
X_sample = np.zeros((data_sample.shape[0], len(vocabulary)))
y_sample = data_sample["spam"].values

# Convert emails to frequency vectors
for i, email in enumerate(data_sample["text"]):
    email_words = email.split()
    for email_word in email_words:
        if email_word.lower() in vocabulary:
            X_sample[i, vocabulary[email_word.lower()]] += 1 #increase the word's frequency in the frequency vector

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
'''
train_test_split takes in these inputs:
    X_sample: feature matrix. Each row is a sample (email freq vector) and each column is a feature (count of a word from vocab)
    y_sample: target vector. Each entry represents a label (whether the email is spam or not)
    test_size: split the data into training and testing sets. 20% will be used for testing, rest training
    random_state: seed for the random number generator. Use any number
    
train_test_split outputs these:
    X_train: subset of the feature matrix for training. Should contain 80% of the original data
    X_test: subset of the feature matrix for testing. Should contain 20% of the original data
    y_train & y_test: labels for training and testing data
'''


# Train Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
