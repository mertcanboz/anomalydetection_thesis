FROM python:3.8
WORKDIR /pipeline
COPY requirement.txt /pipeline
RUN pip3 install -r requirement.txt
COPY fetchData.py /pipeline