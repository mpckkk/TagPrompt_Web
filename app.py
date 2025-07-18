import streamlit as st
import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from io import BytesIO
from wordcloud import WordCloud
from collections import defaultdict
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# ========== NLTK + spaCy Setup ==========
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

def safe_download(resource, path=nltk_data_path):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=path)

for resource in ["punkt", "stopwords", "wordnet"]:
    safe_download(resource)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

tokenizer = TreebankWordTokenizer()

# ========== NLP Processing ==========
def process_text(text, method, lang="english"):
    freq_dict = defaultdict(int)
    port, lanc, wnl = PorterStemmer(), LancasterStemmer(), WordNetLemmatizer()
    lines = text.strip().split('\n')
    stop_words = set(stopwords.words(lang)) if lang in stopwords.fileids() else set()

    for line in lines:
        tokens = tokenizer.tokenize(line.strip())
        tokens = [word.lower() for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stop_words]

        if method == 'PorterStemmer':
            for word in [port.stem(word) for word in tokens]:
                freq_dict[word] += 1
        elif method == 'LancasterStemmer':
            for word in [lanc.stem(word) for word in tokens]:
                freq_dict[word] += 1
        elif method == 'WordNetLemmatizerV':
            for word in [wnl.lemmatize(word, 'v') for word in tokens]:
                freq_dict[word] += 1
        elif method == 'WordNetLemmatizerN':
            for word in [wnl.lemmatize(word, 'n') for word in tokens]:
                freq_dict[word] += 1

    return pd.DataFrame(freq_dict.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)

# ========== Visualization ==========
def generate_bar_plot(df, top_n):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_top = df.head(top_n)
    ax.bar(df_top["Word"], df_top["Frequency"])
    ax.set_title("Top Word Frequencies")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Word")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def generate_wordcloud(df):
    text = ' '.join(df.loc[df['Frequency'] > 0].repeat(df['Frequency'])["Word"])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def extract_entities(text):
    doc = nlp(text)
    data = [(ent.text, ent.label_) for ent in doc.ents]
    return pd.DataFrame(data, columns=["Entity", "Label"]) if data else pd.DataFrame()

# ========== Streamlit UI ==========
st.set_page_config(page_title="TagPrompt+", layout="centered")
st.title("🧠 TagPrompt+: Text Analysis & Visualization Tool")

input_option = st.radio("Input Type", ["Text", "Upload .txt File", "Upload .csv File"])
input_text = ""
csv_df = None

if input_option == "Text":
    input_text = st.text_area("Paste your text here:", height=200)
elif input_option == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
elif input_option == "Upload .csv File":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        csv_df = pd.read_csv(uploaded_file)
        text_cols = [col for col in csv_df.columns if csv_df[col].dtype == "object"]
        if text_cols:
            col_choice = st.selectbox("Choose text column:", text_cols)
            input_text = '\n'.join(csv_df[col_choice].dropna().astype(str))
        else:
            st.warning("No valid text columns found.")

method = st.selectbox("Select Processing Method", [
    "PorterStemmer", "LancasterStemmer", "WordNetLemmatizerV", "WordNetLemmatizerN"
])

lang_choice = st.selectbox("Stopword Language", options=stopwords.fileids())

top_n = st.slider("Top N most frequent words to display:", min_value=5, max_value=50, value=15)

if st.button("🔍 Analyze") and input_text.strip():
    st.info("Processing...")
    df = process_text(input_text, method, lang=lang_choice)
    st.success(f"{len(df)} unique tokens found.")
    st.dataframe(df.head(top_n))

    bar_fig = generate_bar_plot(df, top_n)
    st.pyplot(bar_fig)

    wc_fig = generate_wordcloud(df)
    st.pyplot(wc_fig)

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name=f"{method}_output.csv", mime="text/csv")

    # Bar plot PNG download
    img_bytes = BytesIO()
    bar_fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    st.download_button("📷 Download Bar Plot", data=img_bytes, file_name="bar_plot.png", mime="image/png")

    # Word cloud PNG download
    img_bytes_wc = BytesIO()
    wc_fig.savefig(img_bytes_wc, format='png')
    img_bytes_wc.seek(0)
    st.download_button("🌥 Download Word Cloud", data=img_bytes_wc, file_name="wordcloud.png", mime="image/png")

    # Named Entity Recognition
    with st.expander("🧠 Named Entity Recognition (NER)"):
        ner_df = extract_entities(input_text)
        if not ner_df.empty:
            st.dataframe(ner_df)
        else:
            st.write("No named entities detected.")
