import axios from 'axios';

/*
Component that helps us to query the model api from Fargate container. 
Returns us API status code, the top predictions, along with the next 9 predictions
Each prediction will contain the title, ssoc code and confidence level
*/

const URL = 'https://evening-plateau-95803.herokuapp.com/https://e81tvuwky6.execute-api.us-east-1.amazonaws.com/predict'


const getSSOCData = async (mcfurl) => {
    try {
        const data = await axios.get(URL, {
          params: {
            'mcf_url': mcfurl
          }
          // ,
          // headers: {
          //   'x-api-key': 'ministryofmanpower2022'
          // }
        });
        
        return data;

    } catch (error) {
        throw(error);
    }
}

export default getSSOCData;