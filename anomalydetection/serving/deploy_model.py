import kfp
from kfp import Client

INFERENCE_SERVICE_TEMPLATE = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {inferenceServiceName}
  namespace: {inferenceServiceNamespace}
  annotations:
    sidecar.istio.io/inject: "false"
spec:
  predictor:
    serviceAccountName: minio-sa
    containers:
      - name: kserve-container
        image: pytorch/torchserve:latest-cpu
        args:
          - torchserve
          - --start
          - --no-config-snapshots
          - --model-store=/mnt/models/model-store
          - --ts-config=/mnt/models/config/config.properties
          - --models all
        env:
          - name: STORAGE_URI
            value: {modelStorageUri}
"""


def deploy_pytorch_model(service_name, namespace, storage_uri):
    resource = INFERENCE_SERVICE_TEMPLATE.format(service_name, namespace, storage_uri)
    rop = kfp.dsl.ResourceOp(
        name="create-inference-service",
        k8s_resource=resource,
        action="create",
        attribute_outputs={"name": "{.metadata.name}"}
    )


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(deploy_pytorch_model, __file__ + '.tar.gz')
