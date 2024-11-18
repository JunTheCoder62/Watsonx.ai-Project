import streamlit as st
import os

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

url = "https://jp-tok.ml.cloud.ibm.com"

# Replace with your API key and project ID
watsonx_project_id = "6da063aa-abe7-4d6a-8ceb-0bad8657fdc8"
api_key = "kB_-msYQM4N1dOYAjwHJWQU_laoLaZFZY2QnYvcJ4imB"

def get_model(model_type, max_tokens, min_tokens, decoding):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
    )

    return model

def answer_questions():
    st.title('🌠 Test watsonx.ai LLM')
    st.info('Created by: Juniyara Parisya Setiawan | GEN AI & ML \n\n Source: [GitHub](https://github.com/JunTheCoder62/Watsonx.ai-Project)')
    user_question = st.text_input('Ask a question, for example: What is IBM?')
    st.info('Enter your question and wait for the generated answer.')

    if user_question.strip():
        model_type = ModelTypes.FLAN_UL2
        max_tokens = 100
        min_tokens = 20
        decoding = DecodingMethods.GREEDY

        model = get_model(model_type, max_tokens, min_tokens, decoding)

        # Generate the response
        response = model.generate(prompt=user_question)
        
        # Extract the generated text
        try:
            generated_text = response["results"][0]["generated_text"]
        except (KeyError, IndexError):
            generated_text = "Unable to process the response. Please try again."

        formatted_output = f"""
        **Answer to your question:** {user_question}  
        *{generated_text}*
        """
        st.markdown(formatted_output, unsafe_allow_html=True)

if __name__ == "__main__":
    answer_questions()

import ibm_watson_machine_learning
print(ibm_watson_machine_learning.__version__)
