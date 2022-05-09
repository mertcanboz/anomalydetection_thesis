import kfp
from kfp.v2 import dsl
from kubernetes import client as k8s_client
from kfp import Client
from kfp.v2.dsl import component

#kfserving_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/kfserving/component.yaml')

@dsl.pipeline(
    name='Failure Rate Pipeline', 
    description='PoC Failure Rate Pipeline'
)
def failure_rate_pipeline():
    # Loads the yaml manifest for each component
    fetchData = kfp.components.load_component_from_file('fetchData/fetchData.yaml')
    prepareData = kfp.components.load_component_from_file('prepareData/prepareData.yaml')
    preprocess = kfp.components.load_component_from_file('preprocess/preprocess.yaml')
    train = kfp.components.load_component_from_file('train/train.yaml')
    inference = kfp.components.load_component_from_file('inference/inference.yaml')

    pipeline_conf = kfp.dsl.PipelineConf()
    kfp.dsl.get_pipeline_conf().set_image_pull_secrets([k8s_client.V1ObjectReference(name="artifactory-auth")])
    kfp.dsl.get_pipeline_conf().set_image_pull_policy('Always')

    fetchData_task = fetchData(startyear=2022, startmonth=4, startday=1, starthour=0, endyear=2022, endmonth=4, endday=14, endhour=0)
    fetchData_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # # Run prepare_data task
    # prepareData_task = prepareData(threshold=0.02)
    # prepareData_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    preprocess_task = preprocess(fetchData_task.output)
    preprocess_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_task = train(preprocess_task.output, rounds=100, gridsearch=False)
    train_task.execution_options.caching_strategy.max_cache_staleness = "P0D"


    # kfserving = kfserving_op(action="create",
    #                          model_name="isolation_forest",
    #                          model_uri=train_task.output,
    #                          namespace="kubeflow-user-example-com")
    inference_task = inference(train_task.output)
    inference_task.execution_options.caching_strategy.max_cache_staleness = "P0D"


if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(failure_rate_pipeline, __file__ + '.yaml')
