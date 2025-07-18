import streamlit as st
import nltk
import os
import pandas as pd
from collections import defaultdict
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# ========== Setup NLTK for Streamlit Cloud ==========
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

def safe_download(resource, path=nltk_data_path):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=path)

# Download only valid and necessary NLTK corpora/tokenizers
for resource in ["punkt", "stopwords", "wordnet"]:
    safe_download(resource)

# Use Treebank tokenizer to avoid sent_tokenize entirely
tokenizer = TreebankWordTokenizer()

# ========== NLP Processing Function ==========
def process_text(text, method):
    port_freq, lanc_freq, wnlv_freq, wnln_freq = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    lines = text.strip().split('\n')
    port = PorterStemmer()
    lanc = LancasterStemmer()
    wnl = WordNetLemmatizer()

    for line in lines:
        tokens = tokenizer.tokenize(line.strip())
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

# ========== Streamlit UI ==========
st.set_page_config(page_title="TagPrompt", layout="centered")
st.title("🧠 TagPrompt: Text Stemming & Lemmatization Tool")

input_option = st.radio("Input Type", ["Text", "Upload .txt File"])

input_text = ""
if input_option == "Text":
    input_text = st.text_area("Paste your text here:", height=200)
else:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")

method = st.selectbox("Select Processing Method", [
    "PorterStemmer",
    "LancasterStemmer",
    "WordNetLemmatizerV",
    "WordNetLemmatizerN"
])

if st.button("🔍 Process Text") and input_text.strip():
    st.info("Processing...")
    df = process_text(input_text, method)
    st.success(f"Done! {len(df)} unique tokens found.")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{method}_output.csv",
        mime="text/csv"
    )
