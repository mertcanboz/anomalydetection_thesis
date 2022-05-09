import json
from typing import List, Dict
import logging
import kserve
from kserve import Model

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

class FailureRateTransformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        print("MODEL NAME %s", name)
        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)
        self.timeout = 100

    def preprocess(self, inputs: Dict) -> Dict:
        print(inputs)
        return inputs

    def postprocess(self, inputs: List) -> List:
        print(inputs)
        return inputs