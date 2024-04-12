import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data
data = pd.read_csv('spam-data.csv')

X = data.drop(columns=['Class'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {test_accuracy:.4f}")

# feature vector for the test email
test_email_features = [[40, 1, 9, 0]]

# Test the Email
spam = model.predict(test_email_features)[0]

if pam == 1:
    print("Spam.")
else:
    print("Not spam.")