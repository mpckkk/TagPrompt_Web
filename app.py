import streamlit as st
import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from io import BytesIO

# ========== Setup NLTK for Streamlit Cloud ==========
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

tokenizer = TreebankWordTokenizer()

# ========== Text Processor ==========
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

# ========== Plotting ==========
def generate_bar_plot(df, top_n):
    df = df.sort_values(by="Frequency", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["Word"], df["Frequency"])
    ax.set_title("Top Word Frequencies")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Word")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def generate_wordcloud(df, max_words=100):
    text_freq = {row['Word']: row['Frequency'] for _, row in df.iterrows()}
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words)
    wc.generate_from_frequencies(text_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

# ========== Streamlit UI ==========
st.set_page_config(page_title="TagPrompt", layout="centered")
st.title("🧠 TagPrompt: Text Stemming & Lemmatization Tool")

input_type = st.radio("Choose input type", ["Text", "Upload .txt File", "Upload .csv File"])

input_text = ""
csv_df = None

if input_type == "Text":
    input_text = st.text_area("Paste your text here:", height=200)
elif input_type == "Upload .txt File":
    uploaded_txt = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_txt:
        input_text = uploaded_txt.read().decode("utf-8")
elif input_type == "Upload .csv File":
    uploaded_csv = st.file_uploader("Upload a .csv file", type="csv")
    if uploaded_csv:
        csv_df = pd.read_csv(uploaded_csv)
        st.write("CSV preview:")
        st.dataframe(csv_df.head())
        text_columns = csv_df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            selected_col = st.selectbox("Select text column", text_columns)
            input_text = "\n".join(csv_df[selected_col].dropna().astype(str).tolist())
        else:
            st.warning("No text columns found in uploaded CSV!")

method = st.selectbox("Select Processing Method", [
    "PorterStemmer",
    "LancasterStemmer",
    "WordNetLemmatizerV",
    "WordNetLemmatizerN"
])

top_n = st.slider("Top N most frequent words to display:", min_value=5, max_value=50, value=15)

if st.button("🔍 Process Text") and input_text.strip():
    st.info("Processing...")
    df = process_text(input_text, method)
    df = df.sort_values(by="Frequency", ascending=False)

    st.success(f"{len(df)} unique tokens found.")
    st.dataframe(df.head(top_n))

    # --- Bar Plot ---
    fig_bar = generate_bar_plot(df, top_n)
    st.pyplot(fig_bar)

    # --- Download Bar Plot ---
    img_bar = BytesIO()
    fig_bar.savefig(img_bar, format='png')
    img_bar.seek(0)
    st.download_button("📊 Download Bar Plot", data=img_bar, file_name="bar_plot.png", mime="image/png")

    # --- Word Cloud ---
    fig_wc = generate_wordcloud(df, max_words=top_n)
    st.pyplot(fig_wc)

    img_wc = BytesIO()
    fig_wc.savefig(img_wc, format='png')
    img_wc.seek(0)
    st.download_button("☁️ Download Word Cloud", data=img_wc, file_name="wordcloud.png", mime="image/png")

    # --- Download CSV ---
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name=f"{method}_output.csv", mime="text/csv")
