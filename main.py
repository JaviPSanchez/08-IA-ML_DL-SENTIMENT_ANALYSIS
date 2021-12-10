# ---------LIBRARIES------------ #
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import base64
from PIL import Image
import requests
import yfinance as yf
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()  

import twitter_package.twitter
import google_package.google
import wc_package.wc
import nlp_package.nlp
import kmeans_package.kmeans
# ---------PAGE LAYOUT------------ #

st.set_page_config(page_title='Crypto Dash', layout="wide", initial_sidebar_state="collapsed", page_icon='random')



# ---------PAGE TITLE------------ #

st.markdown("<h1 style='text-align: center; color: black;'>CRYPTO DASH</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.write("")
with col2:
    st.image('./assets/bitcoin.png', caption="Let's do some predictions!", width=400, use_column_width=True)
with col3:
    st.write("")

# ---------SECTION OPTIONS------------ #

col4, col5 = st.columns([2,4])
with col4:
    col4.markdown("<h2 style='text-align: center; color: white; background-color: black; border-radius:1rem; box-shadow: 0 1rem 2rem rgba(0, 0, 0, 0.2); margin:2rem;'>SELECT CRYPTO</h2>", unsafe_allow_html=True)
    tickers = ('BTC-USD', 'ETH-USD', 'BNB-USD', 'USDT-USD', 'SOL-USD', 'ADA-USD', 'USDC-USD', 'XRP-USD', 'DOT-USD', 'LUNA-USD', 'DOGE-USD')
    crypto = col4.multiselect('Select Cryto', tickers)
    start = col4.date_input('Start date', value = pd.to_datetime('2021-01-01'))
    end = col4.date_input('End date', value = pd.to_datetime('today'))
    interval = col4.selectbox('Select Interval', ('1wk', '1h'))
with col5:
    def relativeReturn(df):
        relative = df.pct_change()
        cumulativeReturn = (1+relative).cumprod() - 1
        cumulativeReturn = cumulativeReturn.fillna(0)
        return cumulativeReturn
    if len(crypto) > 0:
        data1 = yf.download(crypto,start,end)['Adj Close']
        data2 = relativeReturn(yf.download(crypto,start,end,interval='1wk')['Adj Close'])
        st.line_chart(data1)
        st.line_chart(data2)


# ---------SECTION APIs------------ #

col6, col7, col8 = st.columns([3,1,3])
with col6:
    twitter_form = st.form("API_TWITTER")
    twitter_form.markdown("<h2 style='text-align: center; width:20rem; color: white;  background-color: black; border-radius:1rem; box-shadow: 0 1rem 2rem rgba(0, 0, 0, 0.2); margin:2rem;'>TWITTER API</h2>", unsafe_allow_html=True)
    twitter_form.image('./assets/twitter.png', caption="Let's do some predictions!",  width=None, use_column_width=True)
    search_term = twitter_form.text_input("Search tweets!")
    limit = twitter_form.slider('Select a range of tweeters',0, 100)
    submit_button = twitter_form.form_submit_button("Search")

    if submit_button:
        # twitter_package.twitter.ApiTwitter(search_term, limit).run_api()
        tweets_api = twitter_package.twitter.ApiTwitter(search_term, limit).run_api()
        st.write(tweets_api)
        # df_tweets = pd.read_csv('api_raw_tweets.csv', index_col=[0])
        # st.write(df_tweets)
        


with col8:
    google_form = st.form("API_GOOGLE")
    google_form.markdown("<h2 style='text-align: center; width:20rem; color: white; background-color: black; border-radius:1rem; box-shadow: 0 1rem 2rem rgba(0, 0, 0, 0.2); margin:2rem;'>GOOGLE API</h2>", unsafe_allow_html=True)
    google_form.image('./assets/google.png', caption="Let's do some predictions!", width=None, use_column_width=True)
    google_form.text_input("Search news!")
    google_form.slider('Select a range of news',0, 100)
    google_form.form_submit_button("Search")

    if submit_button:
        pass
        # google_package.google.main()

# ---------SECTION ANALYSIS------------ #

col7, col8, col9, col10, col11, col12 = st.columns([1,8,1,1,8,1])
with col8:
    uploaded_file_1 = st.file_uploader("Choose a file 1")
    if uploaded_file_1 is not None:
        df_1 = pd.read_csv(uploaded_file_1, index_col=[0])
        st.write(df_1)
        st.write(df_1.shape)
        fig_8 = plt.figure()
        df_1['text'].str.len().plot(kind='hist')
        plt.title('Text lenght')
        st.pyplot(fig_8)

    classifier_name = st.selectbox("Select Classifier", (" ","WORD CLOUD", "NLP-TextBlob", "KMEANS", "RNN"))

   
    if classifier_name == 'WORD CLOUD':  
        wc_package.wc.cloud(df_1)
    elif classifier_name == "NLP":
        nlp_package.nlp.NlpApi(df_1)
    elif classifier_name == "KMEANS":
        kmeans_package.kmeans.Kmeans(df_1)
    # elif classifier_name == "RNN":
    #     st.write('hola')
    else:
        st.write('Choose a Method')

with col11:
    uploaded_file_2 = st.file_uploader("Choose a file 2")
    if uploaded_file_2 is not None:
        df_2 = pd.read_csv(uploaded_file_2, index_col=[0])
        st.write(df_2)
        st.write(df_2.shape)
        fig_20 = plt.figure()
        df_1['text'].str.len().plot(kind='hist')
        plt.title('Text lenght')
        st.pyplot(fig_20)
    classifier_name = st.selectbox("Select Classifier", (" ","WORD CLOUD", "NLP-2", "KMEANS", "RNN"))



