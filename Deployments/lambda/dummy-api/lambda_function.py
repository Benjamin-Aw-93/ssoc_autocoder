import json
import logging
import random
import hashlib
import urllib3
import re

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
    with open('ssoc_desc.json') as json_file:
        ssoc_desc = json.load(json_file)

    # Select one of the dummy data randomly
    data = dummy_data[random.randint(0,49)]

    # Extract the hashed MCF job ad ID from the MyCareersFuture URL
    # Note that we append a "?" to the end of the URL to ensure we capture the ID correctly
    mcf_url = event['queryStringParameters']['mcf_url']
    regex_matches = re.search('\\-{1}([a-z0-9]{32})\\?', mcf_url + "?")

    # Try extracting the MCF job ad ID
    # If not we return an error
    try:
        mcf_job_ad_id_hashed = regex_matches.group(1)
        logger.info('Queried MCF job ad ID (hashed): %s', mcf_job_ad_id_hashed)
    except:
        logger.error('Could not identify hashed MCF job ad ID from the URL.')
        response = {
            'statusCode': 400,
            'body': json.dumps({'Error': 'Invalid URL from MyCareersFuture. Please check that you have a valid URL and call this API again.'})
        }
        return response

    # Retrieve the MCF job ad ID and overwriting it with one of our samples
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
    
    # Appending the SSOC title and description to the output
    for prediction in data['predictions']:
        prediction['SSOC_Title'] = ssoc_desc[prediction['SSOC_Code']]['title']
        prediction['SSOC_Description'] = ssoc_desc[prediction['SSOC_Code']]['description']

    # Compile the output into a dictionary
    output = {
        'mcf_job_id': mcf_job_ad_id,
        'mcf_job_title': mcf_job_title,
        'mcf_job_desc': mcf_job_desc,
        'predictions': data['predictions']
    }

    # Create the response objective
    response = {
        'statusCode': 200,
        'body': json.dumps(output)
    }
    
    logger.info('Completed query!')

    return response