import axios from 'axios';

/*
Component that helps us to query food places within a designated area. 
Can be modified to take in additonal inputs to query other things other than restaurants.  
*/

const URL = 'https://evening-plateau-95803.herokuapp.com/https://d1b3viqczc.execute-api.us-east-1.amazonaws.com/default/dummy-api'


const getSSOCData = async (jobid) => {
    try {
        const data = await axios.get(URL, {
          params: {
            'job_id': jobid
          },
          headers: {
            'x-api-key': 'ministryofmanpower2022'
          }
        });
        
        return data;

    } catch (error) {
        console.log(error);
    }
}

export default getSSOCData;