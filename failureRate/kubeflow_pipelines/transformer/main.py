import argparse
import kserve
from transformer import FailureRateTransformer

import logging
logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

DEFAULT_MODEL_NAME = "failure-rate-model"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--predictor_host",
    help="The URL for the model predict function", required=True
)
parser.add_argument(
    "--model_name", default=DEFAULT_MODEL_NAME,
    help='The name that the model is served under.')

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = FailureRateTransformer(
        name=args.model_name,
        predictor_host=args.predictor_host)
    server = kserve.ModelServer()
    server.start(models=[transformer])