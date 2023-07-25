#!/usr/bin/env python
# coding: utf-8

# Step 0. Unzip enron1.zip into the current directory.

# In[3]:


import zipfile

with zipfile.ZipFile('enron1.zip', 'r') as zip_ref:
    zip_ref.extractall('.')


# Step 1. Traverse the dataset and create a Pandas dataframe. This is already done for you and should run without any errors. You should recognize Pandas from task 1.

# In[4]:


import pandas as pd
import os

def read_spam():
    category = 'spam'
    directory = './enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

ham = read_ham()
spam = read_spam()

df = pd.DataFrame.from_records(ham)
df = df.append(pd.DataFrame.from_records(spam))


# Step 2. Data cleaning is a critical part of machine learning. You and I can recognize that 'Hello' and 'hello' are the same word but a machine does not know this a priori. Therefore, we can 'help' the machine by conducting such normalization steps for it. Write a function `preprocessor` that takes in a string and replaces all non alphabet characters with a space and then lowercases the result.

# In[5]:


import re

def preprocessor(text):
    # Replace all non-alphabetic characters with a space using regex
    normalized_text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert the result to lowercase
    normalized_text = normalized_text.lower()
    
    return normalized_text

# Test the preprocessor function
input_text = "Hello, this is a Test-String 123!@#"
processed_text = preprocessor(input_text)
print(processed_text)


# Step 3. We will now train the machine learning model. All the functions that you will need are imported for you. The instructions explain how the work and hint at which functions to use. You will likely need to refer to the scikit learn documentation to see how exactly to invoke the functions. It will be handy to keep that tab open.

# In[13]:


import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 0: Create a list to store the text data and labels
text_data = []
labels = []

# Step 1: Read the text data and labels from the dataset
data_dir = "enron_data"  # Assuming the subdirectories 'ham' and 'spam' are inside the 'enron_data' directory

# Traversing through subdirectories using os.walk()
for root, _, filenames in os.walk(data_dir):
    for filename in filenames:
        if filename.endswith(".txt"):
            label = os.path.basename(root)  # The subdirectory name will be used as the label
            labels.append(label)
            with open(os.path.join(root, filename), 'r', encoding='latin-1') as file:
                text_data.append(file.read())

# Additional check to ensure that text data is not empty
if len(text_data) == 0:
    raise ValueError("No text data found in the subdirectories. Please check the file names and path.")

# Step 2: Instantiate a CountVectorizer with the preprocessor function
def preprocessor(text):
    # Replace all non-alphabetic characters with a space using regex
    normalized_text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert the result to lowercase
    normalized_text = normalized_text.lower()
    
    return normalized_text

vectorizer = CountVectorizer(preprocessor=preprocessor)

# Step 3: Use train_test_split to split the dataset into a train dataset and a test dataset
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Step 4: Use the vectorizer to transform the existing dataset into a form in which the model can learn from
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 5: Use the LogisticRegression model to fit to the train dataset
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Step 6: Validate that the model has learned something by generating predictions
y_pred = model.predict(X_test_vectorized)

# Step 7: Evaluate the model's performance using the provided functions

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Generate classification report
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)


# Step 4.

# In[14]:


## Step 4: Get the features (words) created by the vectorizer
features = vectorizer.get_feature_names_out()

# Step 5: Get the coefficients (importance) from the model
coefficients = model.coef_[0]

# Step 6: Find the top 10 positive features with the largest magnitude (corresponding to spam)
top_positive_features = sorted(zip(features, coefficients), key=lambda x: x[1], reverse=True)[:10]

# Step 7: Find the top 10 negative features with the largest magnitude (corresponding to ham)
top_negative_features = sorted(zip(features, coefficients), key=lambda x: x[1])[:10]

# Print the top positive and negative features
print("Top 10 Positive Features (Spam):")
for feature, coefficient in top_positive_features:
    print(f"{feature}: {coefficient}")

print("\nTop 10 Negative Features (Ham):")
for feature, coefficient in top_negative_features:
    print(f"{feature}: {coefficient}")


# Submission
# 1. Upload the jupyter notebook to Forage.

# All Done!
