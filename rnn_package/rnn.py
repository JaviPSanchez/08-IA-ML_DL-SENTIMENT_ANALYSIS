import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import streamlit as st

class Rnn:
    def __init__(self, df, api):
        self.df = df.reset_index()
        self.api = api
        self.get_sentiment = self.get_sentiment()
        self.load_pretrained = self.load_pretrained()
        
    def load_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
        model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
        
    def get_sentiment(self):
        if self.api == 'twitter':
            tweets = []
            for i in range(0, 2):
                st.write(self.df.Clean_Tweets[i])
                sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
                tweets.append(sentiment_analysis(self.df.Clean_Tweets[i]))
                st.write(sentiment_analysis(self.df.Clean_Tweets[i]))
            
            
        if self.api == 'google':
            articles = []
            for i in range(0, 2):
                st.write(self.df.title[i])
                sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
                articles.append(sentiment_analysis(self.df.title[i]))
                st.write(sentiment_analysis(self.df.title[i]))
            
            
