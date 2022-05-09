### Requirements
`pip3 install torch-model-archiver` for model archive formation.

`pip3 install torchserve` if you want to debug your process on your local torchserve server.
### Steps to deploy your model
1. Have your pytorch model's state_dict saved via:
   ```
   torch.save(model.state_dict(), '/path/to/save/model.pth')
   ```
2. Prepare a python file containing the model class that implements nn.Module. Check: `model.py`.
3. Prepare a model handler, ideally implementing BaseHandler class from torchserve, in which you implement how to handle the model, i.e. preprocessing, postprocessing etc. Check: `anomaly_detection_handler.py`.
4. Run the command below which basically creates a .zip file containing all the files above, in .mar format. This file is the ultimate form that torchserve will use to serve inference.
   ```
   torch-model-archiver --model-name testsrcnn --handler anomaly_detection_handler.py -v 1.0 --serialized-file=model.pt --model-file model.py
   ```
5. (Optional) if you want to deploy the model on your local server, run:
   ```
   torchserve --start --model-store <path to directory containing your .mar file> --models <model name>=<.mar file name in the directory> --ncs
   ```
   The ``--ncs`` argument is for disabling model snapshot feature.

### Configuration in Min.IO

In order to make torchserve work properly, config.properties file needs to be provided. Check: ``config/config.properties``.
Have a look also at `notebooks/inference-service.ipynb` for basic inference calls via curl.
