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
@st.cache_data
def call_job_description(url):
    ad_descript = query_api(url)
    st.write(ad_descript['job_title'])
    st.components.v1.html(ad_descript['job_desc'], height=600)
@st.cache_data
def compare_similarity(url, sol_df):
    result1 = query_api(url)
    if result1:
        # concatenate the vectors from the API query
        concat_r1 = np.array(result1['embeddings_title'][0]+result1['embeddings_text'][0]).reshape(1, -1)
        # obtain similarity scores
        
        result_df = pd.DataFrame({"SOL Occupation":sol_df['SOL Occupation'],
            'similarity': list(map(lambda x: similarity(concat_r1, np.array(x)).round(3), sol_df['comb']))}).sort_values('similarity', ascending=False)
        return result_df
    else:
        st.error("Error occurred during API call.")
def main():
    st.title("SOL Similarity Checker")
    job_id = st.text_input("Enter MCF Job ID")
    main_url = 'http://localhost:8000/embeddings?id'
    # get job description and display it during the comparison
    descript_url = 'http://localhost:8000/prediction?query_type=id&id'
    # import SOL dataframe and its embeddings
    sol_df = pd.read_csv('data/SOL_embeddings.csv')
    sol_df['emb_title'] = sol_df['emb_title'].apply(literal_eval)
    sol_df['emb_text'] = sol_df['emb_text'].apply(literal_eval)
    sol_df['comb'] = [x + y for x,y in zip(sol_df['emb_title'], sol_df['emb_text'])]
    # split the result across various columns
    col1, col2 = st.columns(2)
    if job_id:
        with col1:
            call_job_description(f'{descript_url}={job_id}')
        with col2:
            sim_df = compare_similarity(f'{main_url}={job_id}', sol_df)
            # output as a filtered list
            value = st.slider("Similarity Filter", 0.0, 1.0, 0.05)
            if value: 
                st.dataframe(sim_df[sim_df['similarity']>value], hide_index=True)
    else:
        st.warning("Please enter the API URL.")

if __name__ == "__main__":
    main()
