import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from merlion.models.anomaly.isolation_forest import IsolationForest
from merlion.utils import TimeSeries
import os
from minio import Minio
import tarfile
from joblib import dump, load

def _inference(args):

    print("start inference")
    # with open(os.path.join(args.data, 'config.json')) as data_file:
    #     config = json.load(data_file)
    # config['model_path'] = os.path.join(args.data, "model.pkl")
    # with open(os.path.join(args.data, 'config.json'), 'w') as out_file:
    #     json.dump(config, out_file)
    # loaded_model = IsolationForest.load(args.data)
    # d = {'value': [0.0003252032520325207]}
    # test_data = pd.DataFrame(data=d, index=[1])
    # test_data = TimeSeries.from_pd(test_data)
    # result = loaded_model.get_anomaly_score(test_data)
    # print(result)

    print(" try Minio client ")
    client = Minio("minio-service.kubeflow.svc.cluster.local:9000", "minio", "minio123", secure=False)
    buckets = client.list_buckets()
    for bucket in buckets:
        print(bucket.name, bucket.creation_date)

    #fetchObject = client.fget_object("mlpipeline", "/artifacts/failure-rate-pipeline-lc2g4/2022/02/18/failure-rate-pipeline-lc2g4-3716809775/train-model-function-Data.tgz", "miniofile")

    fetchObject = client.fget_object("mlpipeline", "/artifacts/failure-rate-pipeline-dknpz/2022/02/28/failure-rate-pipeline-dknpz-1803733580/train-model-function-Data.tgz", "miniofile")

    file = tarfile.open("miniofile")
    file.extractall('/tmp/model')
    file.close()

    print(os.listdir('/tmp/model'))

    skmodel = load('/tmp/model/data/model.pkl')
    print(skmodel)

    # with open(os.path.join('/tmp/model/data', 'config.json')) as data_file:
    #     config = json.load(data_file)
    # config['model_path'] = os.path.join('/tmp/model/data', "model.pkl")
    # with open(os.path.join('/tmp/model/data', 'config.json'), 'w') as out_file:
    #     json.dump(config, out_file)

    # loaded_model_2 = IsolationForest.load('/tmp/model/data')
    # result = loaded_model_2.get_anomaly_score(test_data)
    # print(result)
    # print(" success again")





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    print(args)
    _inference(args)