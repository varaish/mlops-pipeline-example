import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# Load test data
test = pd.read_csv('/opt/ml/processing/test/test.csv')

# Split features and labels
X_test = test.drop(columns=['species'])
y_test = test['species']

# Load model
model = joblib.load('/opt/ml/processing/model/model.joblib')

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

# Save evaluation report
report = {'accuracy': accuracy}
with open('/opt/ml/output/evaluation.json', 'w') as f:
    json.dump(report, f)
