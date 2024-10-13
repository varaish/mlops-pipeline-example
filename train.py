import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load training data
train = pd.read_csv('/opt/ml/processing/train/train.csv')

# Split features and labels
X_train = train.drop(columns=['species'])
y_train = train['species']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join('/opt/ml/model', 'model.joblib')
joblib.dump(model, model_path)
