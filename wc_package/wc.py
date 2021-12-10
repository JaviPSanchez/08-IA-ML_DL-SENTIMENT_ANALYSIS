import matplotlib.pyplot as plt
import wordcloud
import streamlit as st


def cloud(df_1):
    allWords = " ".join([tweets for tweets in df_1['Clean_Tweets']])
    wordcloud1 = wordcloud.WordCloud().generate(allWords)
    fig_4 = plt.figure()
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig_4)
    return fig_4