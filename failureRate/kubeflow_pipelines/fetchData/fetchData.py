import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from prometheus_api_client import *
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime

# this is limited by Protheus, time points cannot exceed this limit per query
MAX_NUMBER_OF_TIME_POINTS = 11000

def _fetch_data(args):

    print(" fetch data from Thanos ")

    prom = PrometheusConnect(url="http://thanos-addons-query.monitoring.svc.cluster.local:9090")
    print(" check thanos connection: ", prom.check_prometheus_connection())

    startTime = datetime(args.startYear, args.startMonth, args.startDay, args.startHour, 0, 0, 0)
    endTime = datetime(args.endYear, args.endMonth, args.endDay, args.endHour, 0, 0, 0)

    metric_df = fetchDataFromThanos(startTime, endTime)

    print(metric_df)

    # dataset = aggregateDataset(metric_df, args.duration)
    # print("start label")
    # print(args.threshold)
    # dataset = labelDataset(dataset, args.threshold)
    # dataset = dataset[['value', 'label']]
    # print(dataset.head())
    # print(dataset.describe())

    # Creates a json object based on `data`
    data_json = metric_df.to_json()
    print(data_json)

    # Saves the json object into a file
    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)

def isTimeRangeExceedLimit(startTime, endTime):
    timeDiff = endTime - startTime
    return timeDiff.total_seconds()/60 > MAX_NUMBER_OF_TIME_POINTS

def queryThanosQueryAPI(startTime, endTime):
    metric_data = prom.custom_query_range(
        query="sum(rate(http_server_requests_seconds_count{job='product-content', outcome!='SUCCESS'}[5m])) / sum(rate(http_server_requests_seconds_count{job='product-content'}[5m]))",
        step = 60,
        start_time=startTime,
        end_time=endTime)

    metric_df = MetricRangeDataFrame(metric_data)
    metric_df = pd.DataFrame(metric_df)
    metric_df.index = pd.to_datetime(metric_df.index, unit='s')

    metric_df['value'] = metric_df['value'].astype('float64')
    return metric_df

def splitTimeRange(startTime, endTime):
    tmpStartTime = startTime
    timeRanges = []
    while tmpStartTime < endTime:
        tmpEndTime = tmpStartTime + timedelta(minutes=MAX_NUMBER_OF_TIME_POINTS) if isTimeRangeExceedLimit(tmpStartTime, endTime) else endTime
        timeRanges.append([tmpStartTime, tmpEndTime])
        tmpStartTime = tmpEndTime + timedelta(minutes=1)
    return timeRanges

def fetchDataFromThanos(startTime, endTime):
    if isTimeRangeExceedLimit(startTime, endTime):
        timeRanges = splitTimeRange(startTime, endTime)
        metricData = pd.DataFrame()
        for timeRange in timeRanges:
            print("time range from %s to %s" %(timeRange[0], timeRange[1]))
            tmpData = queryThanosQueryAPI(timeRange[0], timeRange[1])
            metricData = pd.concat([metricData, tmpData])
        return metricData
    return queryThanosQueryAPI(startTime, endTime)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--startYear', type=int, default=2022)
    parser.add_argument('--startMonth', type=int, default=4)
    parser.add_argument('--startDay', type=int, default=1)
    parser.add_argument('--startHour', type=int, default=0)
    parser.add_argument('--endYear', type=int, default=2022)
    parser.add_argument('--endMonth', type=int, default=4)
    parser.add_argument('--endDay', type=int, default=10)
    parser.add_argument('--endHour', type=int, default=0)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    _fetch_data(args)
    