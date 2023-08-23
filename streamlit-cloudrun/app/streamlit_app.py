""" 
    Streamlit App that calls Vertex AI prediction endpoint
"""

from PIL import Image
import os

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/7756034187169103872"  # <---- CHANGE THIS !!!!
)

st.set_page_config(page_title="Llama 2-7B Chat deployed as Streamlit app")

# Sidebar contents
with st.sidebar:
    st.title('💬 Llama 2 in Vertex AI')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Vertex AI](https://cloud.google.com/vertex-ai)
    - [Llama-2 7B chat](https://ai.meta.com/llama/) LLM model
    
    ''')
    add_vertical_space(5)
    # Logo
    dir_root = os.path.dirname(os.path.abspath(__file__))
    logo = Image.open(dir_root+'/vertexai-logo.webp')
    st.image(logo)

# Generate empty lists for generated and past.
## generated stores AI generated responses
generated = []
past = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm llama-2 7B in Vertex AI, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

## Output
def inference(text):
  response = endpoint.predict([[str(text)]])
  print(response)
  print("separo")
  print(response.predictions[0])
  return str(response.predictions[0])


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = inference(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))