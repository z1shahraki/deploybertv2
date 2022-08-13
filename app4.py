# import packages.data_processor as dp
import streamlit as st
import numpy as np
import joblib
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("ZBH/RSHQBert")
    return tokenizer,model


tokenizer,model = get_model()

# user_input = st.text_area('Enter Text to Analyze')
# button = st.button("Analyze")


d = {'Electricity': 0,
 'Light/other vehicles': 1,
 'Exceedance': 2,
 'Vehicle fire/Fire and Heat': 3,
 'Explosions': 4,
 'Heavy vehicles': 5,
 'Falling or Moving Objects': 6,
 'Fall of Ground': 7,
 'Fall of Person': 8,
 'Machinery': 9,
 'Health - Human': 10,
 'Noise and Vibration': 11,
 'Health - Biological': 12,
 'Fall of Structure': 13}

def main(title = "Your Awesome Text classification App".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 65px; color: #4682B4;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    st.markdown(""" <style> .font {
                font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
                </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> div[role=alert] {background-color: #F5EBFF} 
                </style> """, unsafe_allow_html=True)
    st.image("./images/Sf_3.jpg")
    info = ''
    
    with st.expander("1. Check the category of your text 😀"):
        text_message = st.text_input("Please enter your message")
        if st.button("Predict"):
            test_sample = tokenizer([text_message], padding=True, truncation=True, max_length=512,return_tensors='pt')
            # test_sample
            output = model(**test_sample)
            st.write("Logits: ",output.logits)
            prediction = np.argmax(output.logits.detach().numpy(),axis=1)
            
            #spam_clf.predict(vectorizer.transform([clean_text_message]))
            if(prediction[0] == 0):
                info = 'Electricity'
            elif(prediction[0] == 1):
                info = 'Light/other vehicles'
            elif(prediction[0] == 2):
                info = 'Exceedance'
            elif(prediction[0] == 3):
                info = 'Vehicle fire/Fire and Heat'
            elif(prediction[0] == 4):
                info = 'Explosions'
            elif(prediction[0] == 5):
                info = 'Heavy vehicles'
            elif(prediction[0] == 6):
                info = 'Falling or Moving Objects'
            elif(prediction[0] == 7):
                info = 'Fall of Ground'
            elif(prediction[0] == 8):
                info = 'Fall of Person'
            elif(prediction[0] == 9):
                 info = 'Machinery'
            elif(prediction[0] == 10):
                 info = 'Health - Human'
            elif(prediction[0] == 11):
                 info = 'Noise and Vibration'
            elif(prediction[0] == 12):
                 info = 'ealth - Biological'
            elif(prediction[0] == 13):
                 info = 'Fall of Structure'
            else:
                 info = 'Not recognized'
            st.info('Prediction: {}'.format(info))

if __name__ == "__main__":
    main()