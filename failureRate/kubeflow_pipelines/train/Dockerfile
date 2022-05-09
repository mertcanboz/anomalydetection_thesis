FROM python:3.8
WORKDIR /pipeline
COPY requirement.txt /pipeline
RUN apt-get update && \
    apt-get install -y libgomp1
RUN pip3 install -r requirement.txt
COPY isolation_forest.py /pipeline
COPY train.py /pipeline