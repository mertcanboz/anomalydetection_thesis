import logging
import os
import time

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from model import SRCNN

logger = logging.getLogger(__name__)

WINDOW_SIZE = 1024
THRESHOLD = 0.5


class AnomalyDetectionHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = SRCNN(WINDOW_SIZE)
            self.model.load_state_dict(torch.load(model_pt_path))
            # self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        if type(data.get("body")) is bytearray:
            data = data.get("body").decode()
        data = self.load_kpi(data)[1]
        series = torch.stack(self.load_time_series(data)).to(self.device)
        logger.info('Preprocessed data shape: ' + str(series.shape))
        return series

    def postprocess(self, preds):
        result = preds[0].tolist()
        for pred in preds[1:]:
            result.append(pred.tolist()[-1])
        result = np.asarray(result)
        result[result >= THRESHOLD] = 1
        result[result < THRESHOLD] = 0
        return result.tolist()

    def handle(self, data, context):
        """Entry point for handler. Usually takes the data from the input request and
           returns the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns the data input with the cutout applied.
        """
        start_time = time.time()
        self.context = context
        metrics = self.context.metrics
        data_preprocess = [self.preprocess(datum) for datum in data]
        output = []
        # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        for series in data_preprocess:
            if not self._is_explain():
                inference_result = self.inference(series)
                result = self.postprocess(inference_result)
                output.append(result)
            else:
                # TODO
                output.append(self.explain_handle(series, data))

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        logger.info('Output length: ' + str(len(output[0])))
        return output

    def load_time_series(self, raw_series):
        """
        Provides the data as an array of window size shaped frames
        """

        def normalize(a):
            amin = np.min(a)
            amax = np.max(a)
            a = (a - amin) / (amax - amin + 1e-5)
            return 3 * a

        series = []
        length = len(raw_series)
        for pt in range(WINDOW_SIZE, length, 1):
            head = max(0, pt - WINDOW_SIZE)
            tail = min(length, pt)
            data = np.array(raw_series[head:tail])
            data = data.astype(np.float64)
            data = normalize(data)
            series.append(torch.FloatTensor(data.tolist()))
        return series

    def load_kpi(self, csv_bytearray):
        kpi = [[], []]
        input = str(csv_bytearray).splitlines()
        cnt = 0
        for line in input:
            if not line:
                break
            if cnt == 0:
                cnt += 1
                continue

            row = line.split(',')
            kpi[0].append(int(row[0]))  # timestamp
            kpi[1].append(float(row[1]))  # value
            cnt += 1
        logger.info(f'Time series with length {cnt} loaded.')
        return kpi
