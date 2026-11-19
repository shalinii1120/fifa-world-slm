import re
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------- SETTINGS (edit these) -------------------------
APP_NAME = "FIFA WORLD"
MODEL_ID = "shalinii07/flan-t5-transformer-paper-slm"
DATA_DIR = Path(__file__).parent

def find_csv(name):
    for p in (DATA_DIR / "dataset" / name, DATA_DIR / name):
        if p.exists():
            return p
    return None
THRESHOLD = 0.12          # scope-guard strictness (calibrate in Step 6)
SHOW_SCORE = False        # set True while calibrating
OTHER_PLATFORM = "Google (google.com) or ChatGPT (chatgpt.com)"   # <- change to your platform

WELCOME = f"Hi! Welcome to {APP_NAME} ⚽ I answer questions about the content I was trained on. Go ahead and ask me something!"
OUT_OF_SCOPE = (f"This question is not related to my dataset, so I can't answer it here. "
                f"Please visit {OTHER_PLATFORM} to get the answer.")

GREETINGS = [
    (r"(hi+|hello+|hey+|hiya|hola|namaste)( there| everyone| all)?", WELCOME),
    (r"good (morning|afternoon|evening)( there| everyone| all)?", f"Good day! Welcome to {APP_NAME} ⚽ How can I help you?"),
    (r"good night", f"Good night! Thanks for visiting {APP_NAME}."),
    (r"(how are you|how are you doing|how r u|whats up|what's up|sup)", f"I'm doing great, thanks for asking! Welcome to {APP_NAME}. What would you like to know?"),
    (r"(thanks|thank you|thankyou|thx|ty)( so much| a lot)?", "You're welcome! Ask me anything else about my dataset."),
    (r"(bye|goodbye|see you|see ya|take care)", f"Goodbye! Thanks for visiting {APP_NAME}. 👋"),
    (r"(who are you|what are you|what is your name|your name)", f"I'm {APP_NAME}, a small language model (FLAN-T5-small) fine-tuned on the 'Attention Is All You Need' research paper."),
]

def greeting_reply(text):
    t = re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()
    for pattern, reply in GREETINGS:
        if re.fullmatch(pattern, t):
            return reply
    return None

# ------------------------- Cached loaders -------------------------
@st.cache_resource(show_spinner="Loading the model (first time takes a minute)...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_scope_guard():
    docs = []
    for name in ["train.csv", "validation.csv", "test.csv"]:
        p = find_csv(name)
        if p is not None:
            df = pd.read_csv(p)
            docs += (df["input_text"].astype(str) + " " + df["target_text"].astype(str)).tolist()
    p = find_csv("FIFA_DATASET_TRANSFORMATION.csv")
    if p is not None:
        raw = pd.read_csv(p, encoding="utf-8-sig")
        docs += (raw["section"].fillna("").astype(str) + " " + raw["content"].fillna("").astype(str)).tolist()
    if not docs:
        st.error("No dataset CSV files found in the repository.")
        st.stop()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return vec, vec.fit_transform(docs)

tokenizer, model = load_model()
vec, doc_matrix = load_scope_guard()

def scope_score(q):
    return float(cosine_similarity(vec.transform([q]), doc_matrix).max())

def generate_answer(q):
    inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, num_beams=4,
                             no_repeat_ngram_size=3, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def respond(q):
    g = greeting_reply(q)
    if g:
        return g, None
    score = scope_score(q)
    if score < THRESHOLD:
        return OUT_OF_SCOPE, score
    ans = generate_answer(q)
    return (ans or "I couldn't generate an answer. Please try rephrasing your question."), score

# ------------------------- Page -------------------------
st.set_page_config(page_title=APP_NAME, page_icon="⚽", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #8ecbff 0%, #e6c9ff 100%); }
header[data-testid="stHeader"] { background: transparent; }
.hero { text-align: center; margin-top: 8vh; }
.hero h1 { font-size: 2.6rem; color: #1f2340; margin-bottom: .2rem; }
.hero p  { color: #4a4f73; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(f"""
    <div class="hero">
      <div style="font-size:4rem">⚽</div>
      <h1>{APP_NAME}</h1>
            <p>Ask me about the <b>Attention Is All You Need</b> paper and how it is adapted to <b>FIFA player data</b>.</p>
    </div>
    """, unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask your question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply, score = respond(prompt)
        except Exception as e:
            reply, score = f"Something went wrong: {e}", None
        st.markdown(reply)
        if SHOW_SCORE and score is not None:
            st.caption(f"scope score: {score:.3f} (threshold {THRESHOLD})")
    st.session_state.messages.append({"role": "assistant", "content": reply})
    if len(st.session_state.messages) == 2:
        st.rerun()
