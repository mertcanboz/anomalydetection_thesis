import numpy as np
import pandas as pd
import argparse
import json
from merlion.utils import UnivariateTimeSeries, TimeSeries
from pathlib import Path
from sklearn.model_selection import ParameterGrid, ParameterSampler
from isolation_forest import TyrellIsolationForest
from merlion.post_process.threshold import AggregateAlarms
from merlion.transform.base import Identity
from merlion.evaluate.anomaly import TSADMetric
from merlion.transform.sequence import TransformSequence

def _train_data(args):

    print(args)
    print("start train")
    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)
    dataObj = json.loads(data)
    train_data = pd.read_json(dataObj['train_data'])
    test_data = pd.read_json(dataObj['test_data'])
    test_labels = pd.read_json(dataObj['test_labels'])

    train_data = TimeSeries.from_pd(train_data)
    test_data = TimeSeries.from_pd(test_data)
    test_labels = TimeSeries.from_pd(test_labels)

    param_grid = {
        "enable_preprocess": [True, False],
        "n_estimators": np.arange(50,201,10),
        "alm_threshold": np.arange(1.5,4,0.1),
        "min_alm_in_window": np.arange(0,10,1),
        "alm_window_minutes": np.arange(10,120,10),
        "alm_suppress_minutes": np.arange(10,180,10)
    }
    rng = np.random.RandomState(0)
    print(" rounds: ", args.rounds)
    gridSearchFlag = False
    if args.gridsearch == "True":
        gridSearchFlag = True
    print("choose", "grid search" if gridSearchFlag else "random search")
    param_list = ParameterGrid(param_grid) if gridSearchFlag else list(ParameterSampler(param_grid, n_iter=args.rounds, random_state=rng))
    numberOfGrids = len(param_list)

    best_score = 0
    best_precision = 0
    best_recall = 0
    best_grid = None
    
    for i in np.arange(numberOfGrids):
        #print("parameters:" , param_list[i])
        print(" train loop %s out of %s " %(i, numberOfGrids))
        model = TyrellIsolationForest()
        model.set_params(**param_list[i])
        model.train(train_data)
        f1_score = model.score(test_data, test_labels)['f1']
        print("f1 score:", f1_score)
        # save if having best f1 score
        if f1_score > best_score:
            best_score = f1_score
            best_precision = model.score(test_data, test_labels)['precision']
            best_recall = model.score(test_data, test_labels)['recall']
            best_grid = param_list[i]

    print(" best score: ", best_score)
    print(" best parameters: ", best_grid)
    best_model = TyrellIsolationForest()
    best_model.set_params(**best_grid)
    best_model.train(train_data)
    print(" model performance: ", best_model.score(test_data, test_labels))
    print(args.outputdata)
    best_model.save(args.outputdata)
    print("best model save to Artifact successful ")

    metrics = {
        'metrics': [{
        'name': 'f1-score', # The name of the metric. Visualized as the column name in the runs table.
        'numberValue':  best_score, # The value of the metric. Must be a numeric value.
        'format': "RAW",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        },
        {
        'name': 'precision', # The name of the metric. Visualized as the column name in the runs table.
        'numberValue':  best_precision, # The value of the metric. Must be a numeric value.
        'format': "RAW",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        },
        {
        'name': 'recall', # The name of the metric. Visualized as the column name in the runs table.
        'numberValue':  best_recall, # The value of the metric. Must be a numeric value.
        'format': "RAW",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
    }
    with open(args.metrics, 'w') as f:
        json.dump(metrics, f)
                                     
    # use Merlin IsolationForest in order to save model and load model.
    # config = IsolationForestConfig(transform=TransformSequence([Identity()]),
    #                            n_estimators =100, 
    #                            threshold=AggregateAlarms(alm_threshold=2.2,min_alm_in_window = 2,alm_window_minutes = 10,alm_suppress_minutes = 10))
    # model_merlion  = IsolationForest(config)
    # train_scores_merlion = model_merlion.train(train_data=train_data, anomaly_labels=None)

    # scores_merlion = model_merlion.get_anomaly_score(test_data)
    # labels_merlion = model_merlion.get_anomaly_label(test_data)

    # test_pred = model_merlion.get_anomaly_label(time_series=test_data)

    # p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
    # r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
    # f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
    # mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
    # print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n" f"Mean Time To Detect: {mttd}")

    # print(args.outputdata)
    # model_merlion.save(args.outputdata)
    # print("Merlion model save to Artifact successful ")

if __name__ == '__main__':
    
    print(" haha ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--outputdata', type=str)
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--gridsearch', type=bool)
    args = parser.parse_args()
    Path(args.outputdata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    _train_data(args)