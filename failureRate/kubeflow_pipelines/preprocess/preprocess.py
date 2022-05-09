import numpy as np
import pandas as pd
import argparse
import json
from merlion.utils import UnivariateTimeSeries, TimeSeries
from pathlib import Path

def _preprocess_data(args):

    print("start preprocess")
    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)
    print(data)
    dataset = pd.read_json(data)

    dataset = aggregateDataset(dataset)
    print("start label")
    #print(args.threshold)
    dataset = labelDataset(dataset)
    dataset = dataset[['value', 'label']]
    print(dataset.head())
    print(dataset.describe())


    train_data = TimeSeries.from_pd(dataset['value'][:int(len(dataset)*0.8)])
    test_data = TimeSeries.from_pd(dataset['value'][int(len(dataset)*0.8):])
    test_labels = TimeSeries.from_pd(dataset['label'][int(len(dataset)*0.8):])
    print("length of train data: ", len(train_data))
    print("length of test data: ", len(test_data))
    print("length of test label: ", len(test_labels))

    outputData = { 
                    "train_data": train_data.to_pd().to_json(),
                    "test_data" : test_data.to_pd().to_json(),
                    "test_labels" : test_labels.to_pd().to_json()
                    }

    data_json = json.dumps(outputData)
    # # Creates a json object based on `data`
    # data_json = dataset.to_json()

    # # Saves the json object into a file
    with open(args.outputdata, 'w') as out_file:
        json.dump(data_json, out_file)

def aggregateDataset(df):
    duration = 5
    for i in range(1, duration + 1):
        df['value_' + str(i) ] = df['value'].shift(-i)
    return df

def labelDataset(df):
    threshold=0.02
    print(threshold)
    print(df)
    df['label'] = (df > threshold).all(axis=1)
    print(df)
    df['label'] = df['label'].map({True: 1, False: 0})
    print(df)
    return df

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--outputdata', type=str)
    args = parser.parse_args()
    Path(args.outputdata).parent.mkdir(parents=True, exist_ok=True)
    _preprocess_data(args)
    