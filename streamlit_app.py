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

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Pragya's Portfolio",
        page_icon="👋",
    )

    st.write("# These are demo projects.")

    st.sidebar.success("Select a above demo projects.")

    st.markdown(
        """
        **👈 Select a project demo from the sidebar** to see some examples!
        ### Current Demo Project - 
        - Monkey Pox Prediction [Predictor](https://pragya-projects.streamlit.app/Monkey_Pox_Predictor)
        - Free Translator to English Language [Translator](https://pragya-projects.streamlit.app/Translator)
        - Sentiment Analysis [Sentiment_Analysis](https://pragya-projects.streamlit.app/Sentiment_Analysis)
        - Azure Function App with Web Scraping using BeautifulSoup [Wikipedia Scrape](https://pragya-projects.streamlit.app/Azure_Web_Scrap_Wiki)
        - QnA Chatbot using Meta Hugging Face [ChatBot](https://pragya-projects.streamlit.app/llama2_QnA_Chatbot)
        ### Please check out my profile?
        - [Linkedin](https://www.linkedin.com/in/pragya-bhatia/)
        - [GitHub](https://github.com/pragyabhatia06)
        - [HackerRank](https://www.hackerrank.com/profile/pragya_bhatia_06)
        - [Microsoft Azure Data Fundamentals](https://www.credly.com/badges/aab2fdf4-c454-49fe-9f40-7d1793afbed0/public_url)
        - [Microsoft Azure Fundamentals](https://www.credly.com/badges/7a66102f-5a98-437c-99d2-31744bef91a7/public_url)

    """
    )


if __name__ == "__main__":
    run()
