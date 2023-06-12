import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
import requests
from ast import literal_eval

def query_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def similarity(vec1, vec2):
    return vec1 @ vec2.T/(norm(vec1)*norm(vec2))

def main():
    st.title("SOL Similarity Checker")
    job_id = st.text_input("Enter MCF Job ID")
    main_url = 'http://localhost:8000/embeddings?id'
    # import SOL dataframe and its embeddings
    sol_df = pd.read_csv('data/SOL_embeddings.csv')
    sol_df['emb_title'] = sol_df['emb_title'].apply(literal_eval)
    sol_df['emb_text'] = sol_df['emb_text'].apply(literal_eval)
    sol_df['comb'] = [x + y for x,y in zip(sol_df['emb_title'], sol_df['emb_text'])]
    
    if job_id:
        result1 = query_api(f'{main_url}={job_id}')
        if result1:
            # concatenate the vectors from the API query
            concat_r1 = np.array(result1['embeddings_title'][0]+result1['embeddings_text'][0]).reshape(1, -1)
            # obtain similarity scores
            
            result_df = pd.DataFrame({"SOL Occupation":sol_df['SOL Occupation'],
                'similarity': list(map(lambda x: similarity(concat_r1, np.array(x)).round(3), sol_df['comb']))}).sort_values('similarity', ascending=False)
            # output as a filtered list
            value = st.slider("Select a value", 0.0, 1.0, 0.05)
            if value: 
                st.dataframe(result_df[result_df['similarity']>value])
        else:
            st.error("Error occurred during API call.")
    else:
        st.warning("Please enter the API URL.")

if __name__ == "__main__":
    main()
