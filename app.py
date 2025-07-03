import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict

st.set_page_config(page_title="NASA Climate NLP App", layout="wide")
st.title("🌍 NASA Climate Sentiment & Topics (TF-IDF + KMeans)")

# --- Data loading and preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv("sentiment.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period("M").astype(str)
    return df

@st.cache_resource
def run_topic_model(texts, n_clusters=5):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(texts)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)

    # Top keywords per topic
    top_words = {
        i: [tfidf.get_feature_names_out()[j] for j in model.cluster_centers_[i].argsort()[-10:][::-1]]
        for i in range(n_clusters)
    }

    # Monthly top keywords (optional)
    topic_keywords_by_month = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(texts):
        for topic in range(n_clusters):
            if clusters[i] == topic:
                topic_keywords_by_month[topic][i] = top_words[topic][:3]

    return clusters, top_words

# --- Load data ---
df = load_data()

# --- Sidebar filters ---
with st.sidebar:
    st.header("🔧 Filters")
    selected_sentiments = st.multiselect("Sentiments", df['transformer_sentiment'].unique(), default=list(df['transformer_sentiment'].unique()))
    n_clusters = st.slider("Number of Topics", 2, 10, 5)

# --- Filter + topic model ---
filtered = df[df['transformer_sentiment'].isin(selected_sentiments)].copy()
clusters, top_words = run_topic_model(filtered['clean_text'], n_clusters)
filtered['topic'] = clusters

# --- Topic keywords ---
st.subheader("🧠 Top Keywords by Topic")
for i in range(n_clusters):
    st.markdown(f"**Topic {i}**: {', '.join(top_words[i])}")

# --- Monthly comment volume ---
st.subheader("📊 Monthly Comment Volume")
monthly_volume = df.groupby('month').size().reset_index(name='count')
fig_volume = px.bar(monthly_volume, x='month', y='count', title="Comments per Month")
st.plotly_chart(fig_volume, use_container_width=True)

# --- Topic Trends by Sentiment ---
st.subheader("📈 Topic Trends (Split by Sentiment)")
for sentiment in selected_sentiments:
    st.markdown(f"### {sentiment.capitalize()} Sentiment")
    sub_df = filtered[filtered['transformer_sentiment'] == sentiment]
    trend = sub_df.groupby(['month', 'topic']).size().reset_index(name='count')
    pivot = trend.pivot(index='month', columns='topic', values='count').fillna(0)
    percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig = px.line(percent.reset_index(), x='month', y=percent.columns, markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- Sample comment viewer ---
st.subheader("📝 Sample Comments by Topic")
selected_topic = st.selectbox("Select a Topic", sorted(filtered['topic'].unique()))
samples = filtered[filtered['topic'] == selected_topic][['date', 'clean_text', 'transformer_sentiment']].head(10)
st.dataframe(samples)

# --- Download CSV ---
st.download_button("📥 Download CSV", filtered.to_csv(index=False), file_name="topic_output.csv")
