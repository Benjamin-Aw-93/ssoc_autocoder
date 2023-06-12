import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
import requests
from ast import literal_eval


@st.cache_data
def query_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

@st.cache_data
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
    st.set_page_config(layout="wide")
    st.title("Similarity scoring between MCF job ads and SOL occupations")

    with st.sidebar:
        st.title("Demo App: CO's JobSOLution")
        st.markdown("""Welcome to JobSOLution, a tool for effortlessly comparing MCF job postings to the standardized SOL list! 
        Bid farewell to the tedious manual process of reading and comparing against the SOL list! 
        Simply enter the corresponding MCF job ID, hit Enter, and let JobSOLution handle the rest.""")
        st.subheader("Features:")
        st.markdown("""
        * Natural Language Processing: Preprocesses MCF Job Ads to extract relevant job tasks from job ads
        * Word Embeddings: Converts job tasks into vectors to enable comparison with SOL occupations
        * Similarity Scoring: Calculates cosine similarity scores between MCF job ads and SOL occupations
        """)
        st.subheader("How to locate MCF Job ID:")
        st.image("./data/MCF_ID_example.PNG")


    job_id = st.text_input("Enter MCF Job ID")
    main_url = "http://localhost:8000/embeddings?id"
    desc_url = "http://localhost:8000/prediction?query_type=id&id"
    
    # import SOL dataframe and its embeddings
    sol_df = pd.read_csv('data/SOL_embeddings.csv')
    sol_df['emb_title'] = sol_df['emb_title'].apply(literal_eval)
    sol_df['emb_text'] = sol_df['emb_text'].apply(literal_eval)
    sol_df['comb'] = [x + y for x,y in zip(sol_df['emb_title'], sol_df['emb_text'])]

    sol_detailed_df = pd.read_excel('data/SOL Verification checks.xlsx', sheet_name = 1)

    if job_id:
        
        result1 = query_api(f'{main_url}={job_id}')
        
        if result1:
            # concatenate the vectors from the API query
            concat_r1 = np.array(result1['embeddings_title'][0]+result1['embeddings_text'][0]).reshape(1, -1)
            title_r1 = np.array(result1['embeddings_title'][0]).reshape(1,-1)
            text_r1 = np.array(result1['embeddings_text'][0]).reshape(1,-1)
            
            # obtain similarity scores
            
            result_df = pd.DataFrame({"SOL Occupation":sol_df['SOL Occupation'],
                                      "Combined similarity": list(map(lambda x: similarity(concat_r1, np.array(x)).round(3), sol_df['comb'])),
                                      "Title similarity": list(map(lambda x: similarity(title_r1, np.array(x)).round(3), sol_df['emb_title'])),
                                      "Text similarity": list(map(lambda x: similarity(text_r1, np.array(x)).round(3), sol_df['emb_text']))}).sort_values('Text similarity', ascending=False)
                        
            # output as a filtered list
            value = st.slider("Select a value", 0.0, 1.0, 0.05)
            
            if value: 
                st.dataframe(result_df[result_df['Combined similarity']>value])

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("MCF Job Ad")
                    job_ad = query_api(f'{desc_url}={job_id}')
                    st.markdown(f'##### {job_ad["job_title"]}')
                    st.markdown(job_ad['job_desc'], unsafe_allow_html=True)

                with col2:
                    st.subheader("Top SOL Occupation")
                    top1_SOL_ref_title = result_df["SOL Occupation"].iloc[0]
                    st.markdown(f'##### {top1_SOL_ref_title}')
                    st.markdown(sol_detailed_df[sol_detailed_df['SOL Occupation'] == top1_SOL_ref_title].iloc[0]['Task and Duties'])
                    
        else:
            st.error("Error occurred during API call.")
    else:
        st.warning("Please enter the API URL.")

if __name__ == "__main__":
    main()
