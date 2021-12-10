import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st


class Kmeans:
    def __init__(self, df, api=None):
        self.df = df
        self.kmeans()
        self.find_optimal_clusters()
        self.mini_Batch_Kmeans()
        self.plot_tsne_pca()
        self.get_top_keywords()

    def kmeans(self):
        tfidf = TfidfVectorizer(
        min_df = 5,
        max_df = 0.95,
        max_features = 8000,
        stop_words = 'english'
        )
        tfidf.fit(self.df.Clean_Tweets)
        text = tfidf.transform(self.df.Clean_Tweets)
        return text, tfidf

    def feature_names(self):
        labels = self.tfidf.get_feature_names()
        return labels
    
    def find_optimal_clusters(self, max_k=20):
        iters = range(2, max_k+1, 2)
        sse = []
        for k in iters:
            sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(self.df).inertia_)
            # print('Fit {} clusters'.format(k))
        fig_4, ax = plt.subplots(1, 1)    
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')
        st.pyplot(fig_4)
        return fig_4
    
    def mini_Batch_Kmeans(self):
        clusters = MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(self.text)
        st.write(clusters)
        return clusters

    def plot_tsne_pca(self):
        max_label = max(self.labels)
        max_items = np.random.choice(range(self.df.shape[0]), size=3000, replace=True)
        
        pca = PCA(n_components=2).fit_transform(self.df[max_items,:].todense())
        tsne = TSNE().fit_transform(PCA(n_components=27).fit_transform(self.df[max_items,:].todense()))
        
        idx = np.random.choice(range(pca.shape[0]), size=100, replace=False)
        label_subset = self.labels[max_items]
        label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
        
        fig_6, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
        ax[0].set_title('PCA Cluster Plot')
        ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
        ax[1].set_title('TSNE Cluster Plot')
        st.pyplot(fig_6)
        return fig_6
    

    def get_top_keywords(self,n_terms=10):
        self.df = pd.DataFrame(self.df.Clean_Tweets.todense()).groupby(self.clusters).mean()
        for i,r in self.df.iterrows():
            st.write((f"Cluster: {i}"))
            st.write((','.join([self.labels[t] for t in np.argsort(r)[-n_terms:]])))

