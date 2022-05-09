## KPIs for Anomaly Detection

This document is a placeholder for different KPIs which we can use to evaluate performance of the the choosen anomaly detection models.


- **Domain Knowledge and manually inspect/visualize of results** -- We are doing as of now in our poc and this will continue to happen by default. But If we can fine tune(add labels to subset of data as we very few anomaly points) this method can provide some great numbers for evaluation.
    - **Accuracy** - Simple metric - not that useful as we have very few anomaly points
    - **Minimizing false positives** - Developers' major pain point
    - **Minimizing false negatives**  --> We can avoid missing on any alerts - but can lead to false alarmns
    - A combination of both -- **F1 score?? ROC curve??**


- Synthesize **artificial data** 
    - Use **performance test** framework to create failures/anomalies on stage and dev cluster and evaluate models
        - This will however still require some manual inspection
        - But can generate some real data
        - Does not require any effort - just some performance test runs to overload the system to make some calls fail
    - Other ways to inject/synthesize anomaly data??
        - Manually inject weird data into raw prometheus metrics collected for evaluation
    - Use statistical methods to create data from same distribution
        - Complex

- Manually cause some **service calls to fail??** - Will probably require dev involvement


- Evaluate model on a **different/but similar labelled dataset** --> not super satisfactory but many people in industry use this as validation check
    - Idea is to evaluate model on new dataset after removing labels  and see how it does?
    
- Use **traditional distance based metrics** to compare distance of anomaly points and normal points. 


- **Can clustering metrics be used?** Like cohesion? elbow curves? Inter cluster distances? Silhouette Score? etc. --> I think we can  but need to figure out details, and transformations required for our case as then use some clustering evaluation on top of that. 

Other KPIs which may be relevant at later stage after picking up similar performance algorithms

- **Training time** and resource consumption
    - Time may not be that important at this stage
    - Resource consumption can be deffered as well as we are not evaluating any deep learning models as of now
- **Time to detect/report** --> important - as we do not want to raise alarms/alerts minutes after an issue has occurred


