import pandas as pd
from sklearn.model_selection import train_test_split

# Load Iris dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Split data into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save train and test sets to CSV
train.to_csv('/opt/ml/processing/train/train.csv', index=False)
test.to_csv('/opt/ml/processing/test/test.csv', index=False)
