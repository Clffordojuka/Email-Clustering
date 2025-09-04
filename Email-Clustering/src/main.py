import streamlit as st
import imaplib
import email
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Helper functions from notebook

def get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                charset = part.get_content_charset()
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        if charset:
                            body += payload.decode(charset, errors="replace")
                        else:
                            body += payload.decode(errors="replace")
                    except Exception:
                        body += payload.decode('utf-8', errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset()
            try:
                if charset:
                    body += payload.decode(charset, errors="replace")
                else:
                    body += payload.decode(errors="replace")
            except Exception:
                body += payload.decode('utf-8', errors="replace")
    return body

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = clean_html(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Streamlit UI

st.title("Email Clustering Dashboard")

# Email login section
with st.form("login_form"):
    email_user = st.text_input("Email address")
    email_pass = st.text_input("App password", type="password")
    submitted = st.form_submit_button("Load Emails")

if submitted:
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_user, email_pass)
        mail.select("inbox")
        status, messages = mail.search(None, "ALL")
        email_uids = messages[0].split()
        emails = []
        for uid in email_uids[:50]:  # limit to first 500 emails for responsiveness
            status, msg_data = mail.fetch(uid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            subject = msg["subject"] or ""
            body = get_email_body(msg)
            emails.append(subject + " " + body)
        mail.logout()
        st.success(f"Loaded {len(emails)} emails.")
    except Exception as e:
        st.error(f"Error loading emails: {e}")
        emails = []

if 'emails' in locals() and emails:
    # Preprocessing
    with st.spinner("Preprocessing emails..."):
        cleaned_emails = [preprocess_text(email) for email in emails]

    # Feature extraction
    max_features = st.slider("Max features for TF-IDF", min_value=500, max_value=3000, value=2000, step=100)
    ngram_min = st.number_input("Ngram min range", min_value=1, max_value=3, value=1)
    ngram_max = st.number_input("Ngram max range", min_value=1, max_value=3, value=2)
    max_df = st.slider("Max document frequency", 0.5, 1.0, 0.9)
    min_df = st.slider("Min document frequency", 1, 10, 5)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(ngram_min, ngram_max), max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(cleaned_emails)

    # Clustering
    num_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=5)
    with st.spinner("Clustering emails..."):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        inertia = kmeans.inertia_

    st.write(f"Within-cluster sum of squares (inertia): {inertia}")
    cluster_counts = dict(zip(*np.unique(labels, return_counts=True)))
    cluster_counts = {int(k): int(v) for k, v in cluster_counts.items()}
    st.write("Cluster counts:", cluster_counts)

    # Topic modeling with LDA
    tokenized_emails = [email.split() for email in cleaned_emails]
    dictionary = corpora.Dictionary(tokenized_emails)
    corpus = [dictionary.doc2bow(text) for text in tokenized_emails]
    with st.spinner("Performing topic modeling with LDA..."):
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_clusters, random_state=42, passes=10)
    st.write("LDAtopic keywords:")
    for idx in range(num_clusters):
        top_words = lda_model.show_topic(idx, topn=10)
        topic_keywords = ", ".join([word for word, _ in top_words])
        st.write(f"Topic {idx}: {topic_keywords}")

    # Visualization with t-SNE
    with st.spinner("Visualizing clusters with t-SNE..."):
        tsne = TSNE(n_components=2, perplexity=30, max_iter=2000, random_state=42)
        tsne_results = tsne.fit_transform(X.toarray())

    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10)
    ax.set_title('Email Clusters Visualized with t-SNE')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)