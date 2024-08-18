# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from nltk.corpus import brown

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
nltk.download('vader_lexicon')

def sent_analysis_roberta(text):
    @st.cache_data
    def read_model(): 
        model1 = AutoModelForSequenceClassification.from_pretrained(MODEL)
        return model1
      
    try:
        model = read_model()
        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2]
        }
        st.title("Model cardiffnlp/twitter-roberta-base-sentiment")
        st.write(scores_dict)
    except Exception as ex:
        st.write(ex)
        pass


def sent_analysis_pipeline(text):
    @st.cache_data
    def read_pipeline():   
        try:          
            sent_pipeline1 = pipeline("sentiment-analysis")
            pipelineError = False
        except:
            pipelineError = True
            sent_pipeline1 = ""
            pass
        return sent_pipeline1
       
    try:
        pipelineError = False
        sent_pipeline = read_pipeline()
        if pipelineError == False:
            st.title("Pre-trained sentiment-analysis Transformer Pipeline")
            st.write(sent_pipeline("Everyone try to loves you which is bad"))
    except Exception as ex:
        st.write(str(ex))
        pass
    
def sent_analysis_NLTK(text):
    @st.cache_data
    def nltkmodule():
        try:
            # nltk.download('maxent_ne_chunker')
            sia1 = SentimentIntensityAnalyzer()           
        except Exception as ex:
            st.write(str(ex))
            pass
        return sia1
    sia = nltkmodule()
    out_SA = sia.polarity_scores(text)
    st.title("SentimentIntensityAnalyzer NLTK - vader_lexicon")
    st.write(out_SA)
    
    
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊")
st.markdown("# Enter the sentence to identify the sentiments")
st.write(
    """Kindly Input your Sentence Text in Textbox. """
)

textinp = st.text_input('Please enter the Sentence Text & Click Analysis', 'I am a good Developer')

col1, col2,col3 = st.columns(3)
with col1:
    if st.button("Sentiment Analysis"):
        sent_analysis_NLTK(textinp)
with col2:
    if st.button("Sentiment Analysis Roberta"):
        sent_analysis_roberta(textinp)
with col3:
    if st.button("Transformer Pipeline"):
        sent_analysis_pipeline(textinp)        

