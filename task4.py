import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv('spam-data.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# Split into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predict = model.predict(X_test)

# Print the confusion matrix
predict_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
conmatrix = confusion_matrix(y_test, predict_binary)

print("Confusion Matrix:")
print(conmatrix)