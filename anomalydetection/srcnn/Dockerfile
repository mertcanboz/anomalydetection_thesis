FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ADD srcnn_train /opt/srcnn_train
WORKDIR /opt/srcnn_train

# Add folder for the logs.
RUN mkdir /katib

RUN chgrp -R 0 /opt/srcnn_train \
  && chmod -R g+rwX /opt/srcnn_train \
  && chgrp -R 0 /katib \
  && chmod -R g+rwX /katib

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3", "/opt/srcnn_train/train.py"]
