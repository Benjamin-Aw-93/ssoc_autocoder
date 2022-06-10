from cgi import test
from lambda_function_2 import *
import json
import os 
import pickle
from testfixtures import tempdir
import os, os.path
import glob
import boto3
from moto import mock_s3
import time
from datetime import date,timedelta
import datetime
import pytest


@mock_s3
def test_read_bucket():

    path ='C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/pmp/'
    s3_client = boto3.client('s3')
    test_bucket_name = 'test_bucket'
    test_data = b'col_1,col_2\n1,2\n3,4\n'
    s3_client.create_bucket(Bucket=test_bucket_name,CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})    
    s3_client.put_object(Body=test_data, Bucket=test_bucket_name, Key='processed_test_data.csv')

    s3 = boto3.resource('s3')
    

    df = read_bucket(test_bucket_name,path,'processed_test_data.csv')

    assert len(df.columns) == 2

    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)


@pytest.fixture
def pandas_test_cases():
    path = 'C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/pd_test_cases/MCF_Training_Set_Full .csv'
    df = pd.read_csv(path)
    return df


@mock_s3
def test_task(pandas_test_cases):

    path ='C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/json_write/'

    s3_client = boto3.client('s3')
    test_bucket_name = 'test_bucket'
    s3_client.create_bucket(Bucket=test_bucket_name,CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})   

    task(pandas_test_cases,test_bucket_name,path)

    s3 = boto3.resource('s3')
    buck = s3.Bucket(test_bucket_name)

    count = 0 

    for s3_object in buck.objects.all():
        count+=1 

    assert count == 20

    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)



















