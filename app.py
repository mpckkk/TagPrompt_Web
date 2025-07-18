import streamlit as st
import nltk
import pandas as pd
import os
from collections import defaultdict
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import io

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stemmers and lemmatizer
port = PorterStemmer()
lanc = LancasterStemmer()
wnl = WordNetLemmatizer()

def process_text(text, method):
    port_freq, lanc_freq, wnlv_freq, wnln_freq = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    lines = text.strip().split('\n')

    for line in lines:
        tokens = nltk.word_tokenize(line.strip())
        tokens = [word.lower() for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords.words("english")]

        if method == 'PorterStemmer':
            for word in [port.stem(word) for word in tokens]:
                port_freq[word] += 1
        elif method == 'LancasterStemmer':
            for word in [lanc.stem(word) for word in tokens]:
                lanc_freq[word] += 1
        elif method == 'WordNetLemmatizerV':
            for word in [wnl.lemmatize(word, 'v') for word in tokens]:
                wnlv_freq[word] += 1
        elif method == 'WordNetLemmatizerN':
            for word in [wnl.lemmatize(word, 'n') for word in tokens]:
                wnln_freq[word] += 1

    if method == 'PorterStemmer':
        return pd.DataFrame(port_freq.items(), columns=["Word", "Frequency"])
    elif method == 'LancasterStemmer':
        return pd.DataFrame(lanc_freq.items(), columns=["Word", "Frequency"])
    elif method == 'WordNetLemmatizerV':
        return pd.DataFrame(wnlv_freq.items(), columns=["Word", "Frequency"])
    elif method == 'WordNetLemmatizerN':
        return pd.DataFrame(wnln_freq.items(), columns=["Word", "Frequency"])
    else:
        return pd.DataFrame()

# =========== Streamlit UI ===========
st.set_page_config(page_title="TagPrompt", layout="centered")
st.title("🧠 TagPrompt: Stem & Lemmatize Text")

# Input text or file
input_option = st.radio("Input Type", ["Text", "Text File (.txt)"])

if input_option == "Text":
    input_text = st.text_area("Enter your text:", height=200)
elif input_option == "Text File (.txt)":
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
    else:
        input_text = ""

# Method selection
method = st.selectbox("Select Processing Method", [
    "PorterStemmer", 
    "LancasterStemmer", 
    "WordNetLemmatizerV", 
    "WordNetLemmatizerN"
])

# Process button
if st.button("🔍 Process Text") and input_text:
    st.info("Processing...")
    df = process_text(input_text, method)
    st.success(f"Done! {len(df)} unique tokens found.")
    st.dataframe(df)

    # CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{method}_output.csv",
        mime="text/csv",
    )
