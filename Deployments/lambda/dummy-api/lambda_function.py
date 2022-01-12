import json
import logging
import random
import hashlib
import urllib3

http = urllib3.PoolManager()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Accepts an action and a number, performs the specified action on the number,
    and returns the result.
    :param event: The event dict that contains the parameters sent when the function
                  is invoked.
    :param context: The context in which the function is called.
    :return: The result of the specified action.
    """
    logger.info('Event: %s', event)

    # Reading in the dummy data
    with open('dummy_data.json') as json_file:
        dummy_data = json.load(json_file)

    # Reading in the SSOC titles mapping
    with open('ssoc_titles.json') as json_file:
        ssoc_titles = json.load(json_file)

    # Select one of the dummy data randomly
    data = dummy_data[random.randint(0,9)]

    # Retrieve the MCF job ad ID and overwriting it with one of our samples
    mcf_job_ad_id = event['queryStringParameters']['job_id']
    logger.info('Queried MCF job ad ID: %s', mcf_job_ad_id)
    mcf_job_ad_id = data['MCF_Job_Ad_ID']
    logger.info('Replaced with MCF job ad ID: %s', mcf_job_ad_id)

    # Call the MCF API
    mcf_uuid = hashlib.md5(mcf_job_ad_id.encode()).hexdigest()
    resp = http.request('GET', f'https://api.mycareersfuture.gov.sg/v2/jobs/{mcf_uuid}')
    
    # Error handling if an invalid MCF job ad ID is provided
    if resp.status != 200 or json.loads(resp.data) == {'message': 'UUID is not found in the database.'}:
        response = {
            'statusCode': 400,
            'body': json.dumps({'Error': 'Error calling the MCF Jobs API with the provided MCF job ad ID. Please check that you have a valid MCF job ad ID and call this API again.'})
        }
        return response

    # If not, we extract the job title and description
    mcf_job_title = json.loads(resp.data)['title']
    mcf_job_desc = json.loads(resp.data)['description']
    
    next_9_preds = [(ssoc_pred, ssoc_titles[str(ssoc_pred)]) for ssoc_pred in data['SSOC_5D_Top_10_Preds'][1:10]]

    # Compile the output into a dictionary
    output = {
        'mcf_job_id': mcf_job_ad_id,
        'mcf_job_title': mcf_job_title,
        'mcf_job_desc': mcf_job_desc,
        'correct_ssoc': (data['Correct_SSOC_2020'], ssoc_titles[str(data['Correct_SSOC_2020'])]),
        'top_ssoc_pred': (data['SSOC_5D_Top_10_Preds'][0], ssoc_titles[str(data['SSOC_5D_Top_10_Preds'][0])]),
        'correct_ssoc_proba': data['SSOC_5D_Top_10_Preds_Proba'][0],
        'next_9_preds': next_9_preds,
        'next_9_proba': data['SSOC_5D_Top_10_Preds_Proba'][1:10]
    }

    # Create the response objective
    response = {
        'statusCode': 200,
        'body': json.dumps(output)
    }
    
    logger.info('Completed query!')

    return response