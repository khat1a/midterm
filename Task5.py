import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv('spam-data.csv')
X = data.drop(columns=['label'])
y = data['label']

# logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Load email from emails.txt
with open('emails.txt', 'r') as file:
    email = file.read()

# Extract email
vec = CountVectorizer(vocabulary=X.columns)
features = vec.fit_transform([email])

prediction = model.predict(features)
if prediction[0] == 1:
    print("Spam.")
else:
    print("Not spam.")

# Analyze
analyze = pd.Series(model.coef_[0], index=X.columns)
print("analyze:")
print(analyze)
