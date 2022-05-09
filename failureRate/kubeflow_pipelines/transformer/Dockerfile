FROM python:3.6
WORKDIR /transformer
COPY requirement.txt /transformer
RUN apt-get update && \
    apt-get install -y libgomp1
RUN pip3 install -r requirement.txt
COPY . .
ENTRYPOINT ["python", "-m", "failure_rate_transformer"]