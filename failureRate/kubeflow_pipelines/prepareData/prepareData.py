import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path

def _prepare_data(args):

    print(" load data success ")
    dataset = pd.read_csv('data.csv', index_col="timestamp")
    dataset.index = pd.to_datetime(dataset.index)
    dataset = aggregateDataset(dataset, args.duration)
    print("start label")
    print(args.threshold)
    print("haha")
    dataset = labelDataset(dataset, args.threshold)
    dataset = dataset[['value', 'label']]
    print(dataset.head())
    print(dataset.describe())

    # Creates a json object based on `data`
    data_json = dataset.to_json()

    # Saves the json object into a file
    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)


def aggregateDataset(df, duration=5):
    for i in range(1, duration + 1):
        df['value_' + str(i) ] = df['value'].shift(-i)
    return df

def labelDataset(df, threshold=0.02):
    print(threshold)
    print(df)
    df['label'] = (df > threshold).all(axis=1)
    print(df)
    df['label'] = df['label'].map({True: 1, False: 0})
    print(df)
    return df

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    _prepare_data(args)
    