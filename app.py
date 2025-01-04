import streamlit as st
import re
import pickle
import numpy as np
from nltk.tokenize import SyllableTokenizer


# LOAD THE FILES
@st.cache_resource
def load_model_and_artifacts():
    try:
        with open("svm_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        
        with open("syllable_vocab.pkl", "rb") as vocab_file:
            syllable_vocab = pickle.load(vocab_file)
        
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        
        return model, syllable_vocab, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None


# PREPROCESSING PIPELINE
def preprocess_username(username, syllable_vocab, scaler):
    # step 1 -> remove numbers, punctuation, and underscores
    username_cleaned = re.sub(r"[\d._]+", "", username)

    # step 2 -> tokenize
    tokenizer = SyllableTokenizer()
    tokenized_username = tokenizer.tokenize(username_cleaned)

    # step 3 -> encoding
    binary_vector = [0] * len(syllable_vocab)
    for syllable in tokenized_username:
        if syllable in syllable_vocab:
            index = syllable_vocab.index(syllable)
            binary_vector[index] = 1

    # step 4 -> scaling
    scaled_vector = scaler.transform([binary_vector])
    return scaled_vector


# APP INTERFACE
# custom css for the layout
def custom_css():
    st.markdown(
        """
        <style>
        /* Hide the top navigation bar and hamburger menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Apply Instagram gradient background to the main container */
        .stApp {
            background: linear-gradient(45deg, #f58529, #dd2a7b, #8134af, #515bd4);
            background-size: 400% 400%;
            animation: gradient 10s ease infinite;
            color: white;
        }

        /* Smooth animation for the gradient */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Input box styling - set background to white, text to black */
        textarea, input[type="text"] {
            background-color: #ffffff !important; /* White background */
            color: #000000 !important; /* Black text color */
            border: 1px solid #cccccc !important; /* Light gray border */
            border-radius: 5px !important; /* Rounded corners */
            padding: 10px !important; /* Padding for better spacing */
            caret-color: #000000;  /* Ensure cursor is visible (black) */
        }

        /* Placeholder text styling */
        input::placeholder {
            color: #888888 !important; /* Gray placeholder text */
            opacity: 1 !important; /* Ensure full visibility */
        }

        /* Button styling */
        button {
            background-color: #dd2a7b !important; /* Instagram pink */
            color: white !important;
            border-radius: 10px !important;
            font-weight: bold !important;
        }

        /* Button hover effect */
        button:hover {
            background-color: #f58529 !important; /* Instagram orange */
        }

        /* Header and title text */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }

        /* Center the app content */
        .block-container {
            padding-top: 50px;
            padding-bottom: 50px;
        }

        /* Footer styling */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: rgba(0, 0, 0, 0.7); /* Translucent black background */
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 0.9rem;
            font-family: Arial, sans-serif;
        }

        /* Footer link styling */
        .footer a {
            color: #f58529;
            text-decoration: none;
            font-weight: bold;
        }

        .footer a:hover {
            color: #dd2a7b;
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


custom_css()

st.title("Hong Kong Instagram Username Identifier")
st.markdown("###### This app uses a SVM to predict if an IG user is from Hong Kong solely based on their usernames🚀")

svm_model, syllable_vocab, scaler = load_model_and_artifacts()

if svm_model and syllable_vocab and scaler:
    username_input = st.text_input(label="Enter an Instagram Username", 
                                   placeholder="Enter an Instagram Username",
                                   label_visibility="hidden")

    if username_input:
        preprocessed_input = preprocess_username(username_input, syllable_vocab, scaler)

        prediction = svm_model.predict(preprocessed_input)[0]
        probabilities = svm_model.predict_proba(preprocessed_input)[0]
        
        confidence = probabilities[prediction]*100
        result = "Hong Kong Username 🇭🇰" if prediction == 1 else "Non-Hong Kong Username ❌"

        st.markdown(f"### *{username_input}* is a {result}")
        st.markdown(f"Confidence: **{confidence:.2f}%**")

        st.markdown(
            """
            > Did we get it right? 🤔 Feel free to check out this [repo](https://github.com/kuntiniong/HK-Insta-Identifier) 
            to better understand how the identifier works and help us improve this project! 💡✨  
            """
        )

# footer
st.markdown(
    """
    <div class="footer">
        © 2025 Kun Tin Iong | 
        <a href="https://github.com/kuntiniong/HK-Insta-Identifier-App" target="_blank"> Source Code </a>|
        <a href="https://github.com/kuntiniong" target="_blank"> GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)