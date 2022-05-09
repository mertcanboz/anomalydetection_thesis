## Options for deploying SRE's ML service

This document details research and comparison of different options which we can use to deploy SRE's ML service.


### Flask application

- In short this method involve dockerizing python code for ML service and run docker container within k8s pods
- Simple but not good for long term in terms of scalability, making changes, regular data loading etc. 
https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b
https://cloud.google.com/community/tutorials/kubernetes-ml-ops

### [kubeflow](https://www.kubeflow.org/)
- Natively for k8s
- Open source
- Documentation not that detailed
- no native monitoring features
- Learning curve(might be a little less for us as it has )
- Installation set up and configuration difficult
- Have all functionalities like model tracking, evaluation, teaam collaboration required for ML model deployment(at least based on documentation)
- Backed by Google(some blogs say setup is difficult for other hyperscalers)
- Not  mature
- no support fr yet  for [Feast](https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning)

#### Open issues with kubeflow
https://datarevenue.com/en-blog/kubeflow-not-ready-for-production

#### Tools which complement Kubeflow pipeline deployment and are natively for k8s
- [Seldon] (https://www.kubeflow.org/docs/external-add-ons/serving/seldon/)
    - very easy installation
    - [helps in deployment of k8s pipeline](https://www.seldon.io/seldon-and-kubeflow-1-0/)
    - [Example 1 of E2E pipeline kubeflow+seldon](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html)
    - [Example 2](https://github.com/kubeflow/example-seldon)
- [Arrikto] (https://www.arrikto.com/tutorials/data-science/an-end-to-end-ml-workflow-from-notebook-to-kubeflow-pipelines-with-minikf-kale/)

### [Bodywork](https://bodywork.readthedocs.io/en/latest/)
- Native support for containers, kubernetes
- GitOps based framework for CI/CD support
- Yaml based configuration supported for different ML pipeline steps
- Looks promising with easy setup/configuration and installation



### [Mlflow](https://mlflow.org/)
- Open source light weight 
- Much easy to setup
- Less learning curve
- No native container orchestration support unlike kubeflow
- Easy to setup
- Have all functionalities like model tracking, evaluation, teaam collaboration required for ML model deployment(at least based on documentation)
- Not  mature


#### Resources
https://valohai.com/blog/kubeflow-vs-mlflow/
https://aahansingh.com/mlflow-on-kubernetes
https://www.analyticsvidhya.com/blog/2021/06/mlops-now-made-simple-using-mlflow/


### [Metaflow](https://docs.metaflow.org/)
- Used in Netflix
- Some features not available in opensource version 
- Full capabilities not available for k8s plugin yet (https://github.com/Netflix/metaflow/issues/16) (https://docs.metaflow.org/introduction/roadmap)
- Model tracking, experiment design, ML pipelines etc. features available
- Python friendly and has support for R too


### Generic deployment frameworks(no ML specfic features)
- [Airflow](https://airflow.apache.org/) (mature, offers orchestration for generic software development pipelines, more like worflow manager)
    - but can use [KubernetesPodOperator](https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/operators.html)
- Luigi (more like a pipeline structure for  different tasks, in python)
- ArgoCD 
- Jenkins variants etc.

#### Resources
https://valohai.com/blog/kubeflow-vs-airflow/
https://valohai.com/blog/kubeflow-vs-metaflow/
https://valohai.com/blog/kubeflow-vs-databricks/
https://valohai.com/blog/kubeflow-vs-argo/

### Managed Solutions(offer varietly of functionality but have limited flexibility and are not open source)
- [Amazon Sagemaker](https://aws.amazon.com/sagemaker/)
- [Azure Machine Learning](https://github.com/Azure/AML-Kubernetes)
- [Neptune](https://neptune.ai/blog/mlflow-vs-kubeflow-vs-neptune-differences) (more like centralized metastore) 


#### Resources
https://medium.com/mlearning-ai/mlops-state-2021-bd69165e2e71
https://www.datarevenue.com/en-blog/airflow-vs-luigi-vs-argo-vs-mlflow-vs-kubeflow
