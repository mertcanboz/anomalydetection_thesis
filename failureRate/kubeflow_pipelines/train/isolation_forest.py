from merlion.evaluate.anomaly import TSADMetric
from merlion.post_process.calibrate import AnomScoreCalibrator
from merlion.post_process.threshold import AggregateAlarms
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import Shingle
from merlion.transform.base import Identity
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.utils import UnivariateTimeSeries, TimeSeries
import numpy as np
from joblib import dump, load
import os

class TyrellIsolationForest():
    
        def __init__(self, 
                     n_estimators=100, 
                     alm_threshold=2.0,
                     min_alm_in_window=2,
                     alm_window_minutes=10,
                     alm_suppress_minutes=10,
                     enable_preprocess=True):

            self.n_estimators = n_estimators
            self.alm_threshold = alm_threshold
            self.min_alm_in_window = min_alm_in_window
            self.alm_window_minutes = alm_window_minutes
            self.alm_suppress_minutes = alm_suppress_minutes
            self.enable_preprocess=enable_preprocess
            self.model = self._generate_model()

        def _generate_model(self):
            if self.enable_preprocess == True:
                self.transform = TransformSequence([DifferenceTransform(), Shingle(size=2, stride=1)])
            else:
                self.transform = TransformSequence([Identity()])
            config = IsolationForestConfig(transform=self.transform,
                                            n_estimators = int(self.n_estimators), 
                                            threshold=AggregateAlarms(alm_threshold=self.alm_threshold, 
                                                                      min_alm_in_window = int(self.min_alm_in_window),
                                                                      alm_window_minutes = int(self.alm_window_minutes),
                                                                      alm_suppress_minutes = int(self.alm_suppress_minutes)))
            model = IsolationForest(config)
            return model
        
        def set_params(self, **params):
            if not params:
                return self

            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.kwargs[key] = value
                
            self.model = self._generate_model()
            return self
    
        def train(self, train_data):
            return self.model.train(train_data=train_data, anomaly_labels=None)
        
        def predict(self, X):
            return self.model.get_anomaly_label(X)

        def score(self, test_data, test_labels):
            test_pred = self.model.get_anomaly_label(time_series=test_data)
            p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
            r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
            f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
            mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
            return {'precision' : p, 'recall' : r, 'f1': f1, 'MTTD': mttd}
        
        def save(self, path):
            #dump(self.model, path)
            if not os.path.exists(path):
                os.mkdir(path)
            self.model.save(path)