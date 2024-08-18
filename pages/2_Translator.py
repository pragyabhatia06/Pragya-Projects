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

from transformers import AutoTokenizer,AutoModelWithLMHead
import torch

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-mul-en')

@st.cache_data
def lang_converion_eng(text):
    @st.cache_data
    def read_model():       
        model1 = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-mul-en')
        return model1
    
    try:
        model = read_model()
        # Tokenize the text
        batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])
        # Make sure that the tokenized text does not exceed the maximum
        # allowed size of 512
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["input_ids"] = batch["input_ids"][:, :512]
        batch["attention_mask"] = batch["attention_mask"][:, :512]
        # Perform the translation and decode the output
        translation = model.generate(**batch)
        converted_text = tokenizer.batch_decode(translation, skip_special_tokens=True)     
        st.write('Translated Text is : ', converted_text)
    except Exception as ex:
        st.write(str(ex))
    
    
st.set_page_config(page_title="Free Translation", page_icon="📊")
st.markdown("# Convert Any Langauge to English")
st.write(
    """Kindly Input your Non-English Text in Textbox. """
)

textinp = st.text_input('Please enter the Non-English Text & Click Translate', 'Cześć Proszę wprowadzić zdanie w języku innym niż angielski do przetłumaczenia')


if st.button("Translate"):
    lang_converion_eng(textinp)
