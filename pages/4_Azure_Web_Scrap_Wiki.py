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
import requests

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

st.set_page_config(page_title="Azure Web Scraping", page_icon="📊")
st.markdown("# Azure Web Scraping")
st.write(
    """ #### Azure Function App With Web Scraping using BeautifulSoup & Get Data From wikipedia.
    #### Enter the Detail you want to check for. For Example - London, Project, Weather."""
)

inp = st.text_input("Please enter a word","Weather")

if st.button("Get Information"):
    data = {
        "name": inp
    }
    outdata = requests.post("https://funcapp-azure-web.azurewebsites.net/api/HttpTrigger1",json=data)
    st.write("Find below Information :-")
    st.write_stream(outdata)
