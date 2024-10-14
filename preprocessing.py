import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Argument parser to specify input and output paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-train', type=str, required=True)
    parser.add_argument('--output-test', type=str, required=True)
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.input)

    # Data preprocessing logic (modify based on use-case)
    # Example: dropping missing values, converting categorical features, etc.
    df = df.dropna()
    
    # Split the data into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Save the train and test sets
    train.to_csv(args.output_train, index=False)
    test.to_csv(args.output_test, index=False)
