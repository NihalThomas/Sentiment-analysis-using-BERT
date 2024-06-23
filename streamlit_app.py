import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import validators

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Function to fetch reviews from Yelp
def fetch_reviews(yelp_url):
    if not validators.url(yelp_url):
        return None, "Invalid URL"
    try:
        r = requests.get(yelp_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class': regex})
        reviews = [result.text for result in results]
        if not reviews:
            return None, "No reviews found"
        return reviews, None
    except requests.exceptions.RequestException as e:
        return None, f"An error occurred: {e}"

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
        reviews, error = fetch_reviews(yelp_url)
        
        if error:
            st.error(error)
        else:
            df = pd.DataFrame(np.array(reviews), columns=['review'])
            df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

            st.write(df)
