import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Function to fetch reviews from Yelp
def fetch_reviews(yelp_url):
    r = requests.get(yelp_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]
    return reviews

# Function to perform sentiment analysis
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Streamlit app
st.title('Yelp Review Sentiment Analysis')

# Input for Yelp URL
yelp_url = st.text_input('Enter Yelp URL')

if yelp_url:
    with st.spinner('Fetching and analyzing reviews...'):
        reviews = fetch_reviews(yelp_url)
        
        if reviews:
            df = pd.DataFrame(np.array(reviews), columns=['review'])
            df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

            st.write(df)
        else:
            st.write("No reviews found or invalid URL.")
