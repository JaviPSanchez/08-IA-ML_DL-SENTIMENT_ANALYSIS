from newsapi import NewsApiClient
import os
import pandas as pd
import streamlit as st

class ApiGoogleNews:
    def __init__(self, query, date):
        self.query = query
        self.date_article = date
        self.api_key = os.environ.get('GOOGLE-API-NEWS-KEY')
        self.respond_endpoint = self.results()

    def run_api_google(self):
        self.return_articles = self.results()
        return self.return_articles

    def results(self):
        results_conca = pd.DataFrame(columns=['author', 'date', 'title', 'description', 'url', 'urlToImage','publishedAt','content', 'source.id', 'source.name'])
        newsapi = NewsApiClient(api_key=self.api_key)
        for i in range(1,2):
            results = newsapi.get_everything(q=self.query,sort_by='relevancy', language='en', page=i, from_param=self.date_article)
            results_conca = results_conca.append(pd.json_normalize(results["articles"]))
        results_conca.to_csv('./assets/database/raw_google_results.csv')
        return results_conca

    


    


