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

import joblib
import os
from sklearn import preprocessing
import pickle
from sklearn.metrics import __all__
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression



def predict_MP():
    @st.cache_data
    def read_model():
        modelname = "Linear_Regression_Model_MP.pkl"
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        
        loaded_model = pickle.load(open(parent_dir + "/model/" + modelname, 'rb'))
        return loaded_model
    
    dictmap = {"None":0,"Fever":1,"Muscle Aches and Pain":2,"Swollen Lymph Nodes":3}
    mapping = {"No": 0, "Yes": 1}
    
    symptomslist = [HIV_Infection,Rectal_Pain,Sexually_Transmitted_Infection, dictmap[Systemic_Illness],Penile_Oedema, Sore_Throat,Solitary_Lesion,Swollen_Tonsils]
    
    symptomslist_encoded = [mapping.get(item, item) for item in symptomslist]
    
    loaded_model_pkl = read_model()

    out = loaded_model_pkl.predict(pd.DataFrame([symptomslist_encoded]))
    if int(out) == 0:
        st.write("Negative")
    else:
        st.write("Positive")
    
st.set_page_config(page_title="Monkey Pox Prediction", page_icon="📊")
st.markdown("# Monkey Pox Prediction")
st.write(
    """Kindly provide your symptom to check whether you have a Monkey Pox. """
)

Systemic_Illness = st.selectbox("Systemic Illness",["None","Fever","Swollen Lymph Nodes","Muscle Aches and Pain"]) 
col1, col2,col3 = st.columns(3)
with col1:
    Sore_Throat = st.selectbox("Sore Throat",["No","Yes"])
with col2:
    Swollen_Tonsils = st.selectbox("Swollen Tonsils",["No","Yes"])
with col3:
    HIV_Infection = st.selectbox("HIV Infection",["No","Yes"])
coln1, coln2 = st.columns(2)
with coln1:
    Rectal_Pain = st.selectbox("Rectal Pain",["No","Yes"])
with coln2:
    Sexually_Transmitted_Infection = st.selectbox("Sexually Transmitted Infection",["No","Yes"])
cl1, cl2 = st.columns(2)
with cl1:
    Penile_Oedema	 = st.selectbox("Penile Oedema",["No","Yes"])
with cl2:
    Solitary_Lesion = st.selectbox("Solitary Lesion",["No","Yes"])

if st.button("Predict"):
    predict_MP()
