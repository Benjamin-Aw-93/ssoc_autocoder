from asyncore import write
from tabnanny import check
import pytest
from smart_open import s3_iter_bucket
from lambda_function import *
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


@pytest.fixture
def integration_test_cases():
    with open('integration_test_cases.json') as f:
        integration_test_cases = json.load(f)
    return integration_test_cases

def test_process_text(integration_test_cases):

    for test_case in integration_test_cases:
        assert cleaning_text_and_check(test_case['input']) == test_case['output']


# Loading json test answer json files as a pytest fixture
@pytest.fixture
def json_lst():

    lst = []

    for filename in os.listdir("json_test/Ans"):
        f = open("json_test/Ans/" + filename)
        entry = json.load(f)
        lst.append(entry)

    return lst


# Loading json test solutuion pickle files as a pytest fixture
@pytest.fixture
def json_ans():

    lst = []

    for filename in os.listdir("json_test/Sol"):
        f = open("json_test/Sol/" + filename, 'rb')
        entry = pickle.load(f)
        lst.append(entry)

    return lst


def test_extract_mcf_data(json_lst, json_ans):

    # Testing different variations, each with different missing information (1)
    assert extract_mcf_data(json_lst[0])[0] == json_ans[0]

    # Testing different variations, each with different missing information (2)
    assert extract_mcf_data(json_lst[1])[0] == json_ans[1]

    # Testing different variations, each with different missing information (3)
    assert extract_mcf_data(json_lst[3])[0] == json_ans[3]

    # Testing different variations, each with different missing information (4)
    assert extract_mcf_data(json_lst[4])[0] == json_ans[4]

    # Testing different variations, each with different missing information (5)
    assert extract_mcf_data(json_lst[5])[0] == json_ans[5]

    # Testing different variations, each with different missing information (6)
    assert extract_mcf_data(json_lst[6])[0] == json_ans[6]

    # Testing different variations, each with different missing information (7)
    assert extract_mcf_data(json_lst[2])[0] == json_ans[2]

    # Check if date is extracted corrected
    assert extract_mcf_data(json_lst[0])[1] == "2018-12-20"



def test_create_file(tmpdir):
    p = tmpdir.mkdir("tmp")
    

# test write to csv
@pytest.fixture
def first_entry():
    data = {'2021-01':[{'uuid': '539df02ed701baaa6d20b5e8fb82b9cc', 'title': 'Sales director', 'description': '<h3>Responsibilities:</h3>\n<h3>· Lead the management team and being responsible for the APAC partnerships</h3>\n<h3>· Growing the revenue through orchestrating execution and strategic planning across different channels of clients</h3>\n<h3>· Ability to leverage across Business to Business clients and targeting the cross functionally – offering exemplary customer service support for the food safety clients across the region</h3>\n<h3>· Growth of the services and education portfolio through key customers</h3>\n<h3>Requirements:</h3>\n<h3>· Minimum 10 years in the food industry</h3>\n<h3>· Track record of success across business development and marketing across the region</h3>\n<h3>· Must have excellent communication skills as well as sensitivity towards all markets</h3>\n<h3>· Strong negotiations and ability to strategise and always see further opportunities across the region</h3>\n<h3>· Ability to travel domestically and internationally</h3>', 'minimumYearsExperience': 10, 'numberOfVacancies': 1, 'skills': 'Account Management, Strategic Planning, Sales, Leadership, Food Safety, Sales Management, Partnership, Food Industry, Targeting, Strategy, business to business clients, New Business Development, Business Development, customer service support, Management', 'company_name': 'ARGYLL SCOTT CONSULTING PTE. LTD.', 'company_description': None, 'company_ssicCode': '70201', 'company_employeeCount': None, 'originalPostingDate': '2020-01-07', 'newPostingDate': '2020-01-07', 'expiryDate': '2020-01-21', 'totalNumberOfView': 57, 'totalNumberJobApplication': 7, 'salary_maximum': 15000, 'salary_minimum': 12000, 'salary_type_id': 4, 'salary_type': 'Monthly', 'MCF_Job_Ad_ID': 'MCF-2020-0004327', 'JSON to CSV date': datetime.datetime(2022, 6, 2, 15, 32, 2, 956494)}, {'uuid': '025e20c4553746731c77858dde113dcb', 'title': 'E&I INSPECTOR (COMPEX)', 'description': '<h3>· Involved in Inspection and Commissioning activities, review technical drawings and Support with Engineering for the technical issues.</h3>\n<h3>· Supervise Instrument Commissioning team, Plans and Conducts Commissioning activities in the assigned area and co-ordinates these activities with operation and other Commissioning Sections efficiently and safely.</h3>\n<h3>· Supervise the Instrument Commissioning Team Conducting Field Commissioning activities for a wide variety of Instrumentation and Control systems such as Field transmitters, Control valves, and DCS, PLC and ESD systems.</h3>\n<h3>· Assure that all components in the Loop meet Specification &amp; Calibration requirements.</h3>\n<h3>· Verify action of Interlocks &amp; Supervise simulation operations on the Instruments, correct interventions off alarms.</h3>\n<h3>· Verify right operation of Loop including Fail-safe Valve Actions.</h3>\n<h3>· Making as Build Mark-up Drawings &amp; Summiting to Clients.</h3>\n<h3>· Co-ordinate with other commissioning Supervisor and System Engineers in Planning and Completing Commissioning activities.</h3>\n<h3>· Preparing Site Reports &amp; Job completion certificates</h3>', 'minimumYearsExperience': 5, 'numberOfVacancies': 5, 'skills': 'Document Management, Operation, Engineering Management, Project Control, Project Management, Engineering, Inspection, Instrumentation, technical issues, Electrical Engineering', 'company_name': 'LISOON INDUSTRIAL ENGINEERING PTE. LTD.', 'company_description': '<p>LISOON INDUSTRIAL ENGINEERING PTE. LTD.</p>\r\n', 'company_ssicCode': '30110', 'company_employeeCount': None, 'originalPostingDate': '2020-01-08', 'newPostingDate': '2020-01-08', 'expiryDate': '2020-02-07', 'totalNumberOfView': 137, 'totalNumberJobApplication': 7, 'salary_maximum': 3200, 'salary_minimum': 2500, 'salary_type_id': 4, 'salary_type': 'Monthly', 'MCF_Job_Ad_ID': 'MCF-2020-0005420', 'JSON to CSV date': datetime.datetime(2022, 6, 2, 15, 32, 2, 956494)}]}
    return data

@pytest.fixture
def second_entry():

    data = {'2021-01':[{'uuid': '539df02ed701baaa6d20b5e8fb82b9cc', 'title': 'Sales director', 'description': '<h3>Responsibilities:</h3>\n<h3>· Lead the management team and being responsible for the APAC partnerships</h3>\n<h3>· Growing the revenue through orchestrating execution and strategic planning across different channels of clients</h3>\n<h3>· Ability to leverage across Business to Business clients and targeting the cross functionally – offering exemplary customer service support for the food safety clients across the region</h3>\n<h3>· Growth of the services and education portfolio through key customers</h3>\n<h3>Requirements:</h3>\n<h3>· Minimum 10 years in the food industry</h3>\n<h3>· Track record of success across business development and marketing across the region</h3>\n<h3>· Must have excellent communication skills as well as sensitivity towards all markets</h3>\n<h3>· Strong negotiations and ability to strategise and always see further opportunities across the region</h3>\n<h3>· Ability to travel domestically and internationally</h3>', 'minimumYearsExperience': 10, 'numberOfVacancies': 1, 'skills': 'Account Management, Strategic Planning, Sales, Leadership, Food Safety, Sales Management, Partnership, Food Industry, Targeting, Strategy, business to business clients, New Business Development, Business Development, customer service support, Management', 'company_name': 'ARGYLL SCOTT CONSULTING PTE. LTD.', 'company_description': None, 'company_ssicCode': '70201', 'company_employeeCount': None, 'originalPostingDate': '2020-01-07', 'newPostingDate': '2020-01-07', 'expiryDate': '2020-01-21', 'totalNumberOfView': 57, 'totalNumberJobApplication': 7, 'salary_maximum': 15000, 'salary_minimum': 12000, 'salary_type_id': 4, 'salary_type': 'Monthly', 'MCF_Job_Ad_ID': 'MCF-2020-0004327', 'JSON to CSV date': datetime.datetime(2022, 6, 2, 15, 32, 2, 956494)}, {'uuid': '025e20c4553746731c77858dde113dcb', 'title': 'E&I INSPECTOR (COMPEX)', 'description': '<h3>· Involved in Inspection and Commissioning activities, review technical drawings and Support with Engineering for the technical issues.</h3>\n<h3>· Supervise Instrument Commissioning team, Plans and Conducts Commissioning activities in the assigned area and co-ordinates these activities with operation and other Commissioning Sections efficiently and safely.</h3>\n<h3>· Supervise the Instrument Commissioning Team Conducting Field Commissioning activities for a wide variety of Instrumentation and Control systems such as Field transmitters, Control valves, and DCS, PLC and ESD systems.</h3>\n<h3>· Assure that all components in the Loop meet Specification &amp; Calibration requirements.</h3>\n<h3>· Verify action of Interlocks &amp; Supervise simulation operations on the Instruments, correct interventions off alarms.</h3>\n<h3>· Verify right operation of Loop including Fail-safe Valve Actions.</h3>\n<h3>· Making as Build Mark-up Drawings &amp; Summiting to Clients.</h3>\n<h3>· Co-ordinate with other commissioning Supervisor and System Engineers in Planning and Completing Commissioning activities.</h3>\n<h3>· Preparing Site Reports &amp; Job completion certificates</h3>', 'minimumYearsExperience': 5, 'numberOfVacancies': 5, 'skills': 'Document Management, Operation, Engineering Management, Project Control, Project Management, Engineering, Inspection, Instrumentation, technical issues, Electrical Engineering', 'company_name': 'LISOON INDUSTRIAL ENGINEERING PTE. LTD.', 'company_description': '<p>LISOON INDUSTRIAL ENGINEERING PTE. LTD.</p>\r\n', 'company_ssicCode': '30110', 'company_employeeCount': None, 'originalPostingDate': '2020-01-08', 'newPostingDate': '2020-01-08', 'expiryDate': '2020-02-07', 'totalNumberOfView': 137, 'totalNumberJobApplication': 7, 'salary_maximum': 3200, 'salary_minimum': 2500, 'salary_type_id': 4, 'salary_type': 'Monthly', 'MCF_Job_Ad_ID': 'MCF-2020-0005420', 'JSON to CSV date': datetime.datetime(2022, 6, 2, 15, 32, 2, 956494)}]
    ,'2020-01':[]}
    return data

# test if it can be written for one file or empty
def test_write_to_csv(first_entry):
    path = 'C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/pmp'
    os.chdir(path)
    try:

        write_to_csv(path,first_entry)

    except Exception as e:
        print(e)

    # test if the file was written 
    assert len([name for name in os.listdir(path) if os.path.isfile(name)]) == 1
    
    #removing all the files in the directory
    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)

    try:

        write_to_csv(path,{})

    except Exception as e:
        print(e)

    # test if the error is thrown for empty file 
    assert len([name for name in os.listdir(path) if os.path.isfile(name)]) == 0
    
    #removing all the files in the directory
    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)

# test for more than one file
def test_write_to_csv(second_entry):
    path = 'C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/pmp'
    os.chdir(path)
    try:

        write_to_csv(path,second_entry)

    except Exception as e:
        print(e)

    # test if the file was written 
    assert len([name for name in os.listdir(path) if os.path.isfile(name)]) == 2

    #removing all the files in the directory
    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)

    

# test output the processed files
@pytest.fixture
def test_raw_csv():
    path = 'C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/test_raw_csv/'
    return path


def test_output_individual_files(test_raw_csv):

    path = test_raw_csv

    orig_length = len([name for name in os.listdir(path) if os.path.isfile(name)])

    output_individual_files(path)

    # test if the files was written successfully
    assert len([name for name in os.listdir(path) if os.path.isfile(name)]) == orig_length*2

    #removing all the processed files in the directory
    filelist = glob.glob(os.path.join(path,'processed*'))
    for f in filelist:

        os.remove(f)



@mock_s3
def test_get_object():


    #initilizing mock s3
    s3_client = boto3.client('s3')
    test_bucket_name = 'test_bucket'
    test_data = b'col_1,col_2\n1,2\n3,4\n'

    s3_client.create_bucket(Bucket=test_bucket_name,CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})    
    s3_client.put_object(Body=test_data, Bucket=test_bucket_name, Key=f'example/s3/path/key/test_data.csv')

    #wait for 10s after the file is uploaded
    time.sleep(3)

    s3 = boto3.resource('s3')
    
    buck=s3.Bucket(test_bucket_name)
    #putting 1second
    time1 = 1 
    yesterday = datetime.datetime.fromisoformat(str(datetime.datetime.now() - timedelta(seconds=time1)))
    
            
    for s3_object in buck.objects.all():

        assert get_object(s3_object,yesterday) == False


@mock_s3
def test_read_buck():
    #initilizing mock s3
    path = 'C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/pmp/'
    s3_client = boto3.client('s3')
    test_bucket_name = 'test_bucket'
    test_data = b'col_1,col_2\n1,2\n3,4\n'
    s3_client.create_bucket(Bucket=test_bucket_name,CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})    
    s3_client.put_object(Body=test_data, Bucket=test_bucket_name, Key='test_data.csv')

    time = 1
    yesterday = datetime.datetime.fromisoformat(str(datetime.date.today() - timedelta(days=time)))
    s3 = boto3.resource('s3')
    buck=s3.Bucket(test_bucket_name)


    read_buck(buck,yesterday,path)
    count = 0 
    for filename in os.listdir(path):
        count+=1 

    assert count ==1 

    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)

    





@mock_s3
def test_wrtie_buck():
    #initilizing mock s3

    path ='C:/Users/Hari Shiman/Documents/SSOC encoder/ssoc_autocoder-master/Notebooks/pmp/'
    s3_client = boto3.client('s3')
    test_bucket_name = 'test_bucket'
    test_data = b'col_1,col_2\n1,2\n3,4\n'
    s3_client.create_bucket(Bucket=test_bucket_name,CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})    
    s3_client.put_object(Body=test_data, Bucket=test_bucket_name, Key='processed_test_data.csv')

    time = 1
    yesterday = datetime.datetime.fromisoformat(str(datetime.date.today() - timedelta(days=time)))
    s3 = boto3.resource('s3')
    buck=s3.Bucket(test_bucket_name)


    read_buck(buck,yesterday,path)

    directed_buck = 'test_buck'
    s3_client.create_bucket(Bucket=directed_buck,CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})  

    write_buck(directed_buck,path)

    filelist = glob.glob(os.path.join(path,'*'))
    for f in filelist:
        os.remove(f)

    buck_2=s3.Bucket(directed_buck)

    count = 0 

    for s3_object in buck_2.objects.all():
        print(s3_object)
        count+=1

    assert count == 1 






    


if __name__ == '__main__':
    pytest.main()

