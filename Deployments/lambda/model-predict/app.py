import json
import random
import hashlib
import urllib3
import re

http = urllib3.PoolManager()

import model_prediction

def handler(event, context):

    # Extract the hashed MCF job ad ID from the MyCareersFuture URL
    print("Reading in MCF URL")
    mcf_url = event['queryStringParameters']['mcf_url']

    # Check if the user wants to randomly grab a previous query
    if mcf_url == 'feelinglucky':
        
        print("User wants a random previous query")

        # Reading in the dummy data
        with open('dummy_data.json') as json_file:
            dummy_data = json.load(json_file)

        # Select one of the dummy data randomly
        data = dummy_data[random.randint(0,49)]

        # Retrieve the MCF job ad ID and overwriting it with one of our samples
        mcf_job_ad_id = data['MCF_Job_Ad_ID']
        mcf_uuid = hashlib.md5(mcf_job_ad_id.encode()).hexdigest()

        print("Random previous query fetched successfully")

    # Otherwise, assume user wants to generate prediction for a new job ad
    else:

        print("User wants to generate prediction for a new MCF job ad")

        # Note that we append a "?" to the end of the URL to ensure we capture the ID correctly
        regex_matches = re.search('\\-{1}([a-z0-9]{32})\\?', mcf_url + "?")

        # Try extracting the MCF job ad ID from the URL
        try:
            mcf_uuid = regex_matches.group(1)
            print(f'Queried MCF job ad ID (hashed): {mcf_uuid}')
        except:
            print('Could not identify hashed MCF job ad ID from the URL.')
            response = {
                'statusCode': 400,
                'body': json.dumps({'Error': 'Invalid URL from MyCareersFuture. Please check that you have a valid URL and call this API again.'})
            }
            return response
    
    # Call the MCF API
    resp = http.request('GET', f'https://api.mycareersfuture.gov.sg/v2/jobs/{mcf_uuid}')

    # Error handling if an invalid MCF job ad ID is provided
    if resp.status != 200 or json.loads(resp.data) == {'message': 'UUID is not found in the database.'}:
        response = {
            'statusCode': 400,
            'body': json.dumps({'Error': 'Error calling the MCF Jobs API with the provided MCF URL. Please check that you have a valid MCF URL and call this API again.'})
        }
        return response

    print('MCF API called successfully')

    # If not, we extract the job title and description
    mcf_job_ad_id = json.loads(resp.data)['metadata']['jobPostId']
    mcf_job_title = json.loads(resp.data)['title']
    mcf_job_desc = json.loads(resp.data)['description']

    # Reading in the SSOC titles mapping
    with open('artifacts/ssoc_desc.json') as json_file:
        ssoc_desc = json.load(json_file)

    # Generate predictions
    if mcf_url == 'feelinglucky':

        # Appending the SSOC title and description to the output
        for prediction in data['predictions']:
            prediction['SSOC_Title'] = ssoc_desc[prediction['SSOC_Code']]['title']
            prediction['SSOC_Description'] = ssoc_desc[prediction['SSOC_Code']]['description']

        # Assigning it to the final predictions object
        predictions_formatted = data['predictions']

        print('Formatted previous query correctly for output')

    else:

        print('Loading custom neural network and generating prediction')

        predictions = model_prediction.model_predict('artifacts/ssoc-autocoder-model.pickle', 
                                                    'artifacts/distilbert-tokenizer-pretrained-7epoch', 
                                                    'artifacts/ssoc-idx-encoding.json', 
                                                    mcf_job_title,
                                                    mcf_job_desc)['SSOC_5D']

        predictions_formatted = []
        for i in range(0, 10):
            prediction_formatted = {
                'SSOC_Code': str(predictions['predicted_ssoc'][i]),
                'SSOC_Title': ssoc_desc[str(predictions['predicted_ssoc'][i])]['title'],
                'SSOC_Description': ssoc_desc[str(predictions['predicted_ssoc'][i])]['description'],
                'Prediction_Confidence': f"{predictions['predicted_proba'][i]*100:.2f}%"
            }
            predictions_formatted.append(prediction_formatted)

        print('Formatted predictions correctly for output')

    # Initialise output with the MCF information
    output = {
        'mcf_job_id': mcf_job_ad_id,
        'mcf_job_title': mcf_job_title,
        'mcf_job_desc': mcf_job_desc,
        'top_prediction': predictions_formatted[0],
        'other_predictions': predictions_formatted[1:10]
    }

    print("Output body generated successfully")

    # Create the response object
    response = {
        'statusCode': 200,
        'body': json.dumps(output)
    }

    print('Script completed successfully!')
    print(output)

    return response

# if __name__ == "__main__":
#     event = {
#         'queryStringParameters': {
#             'mcf_url': 'https://www.mycareersfuture.gov.sg/job/public/data-scientist-government-technology-agency-d4beb5aee362d4d7d340abdd4ea63d7a'
#         }
#     }
#     handler(event, context = '')