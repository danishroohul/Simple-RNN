import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb


# Load the model
model = load_model('simple_rnn.h5')

# Helper function to decode the input and padding
reverse_word_index = {v:k for k,v in imdb.get_word_index().items()}
def decode_review(encoded_review):
    return " ".join([reverse_word_index[word] for word in encoded_review])

def preprocess_input(text):
    words = text.lower().split()
    encoded_review = [imdb.get_word_index()[word] for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Prediction function
def predict_sentiment(review):
    processed_review = preprocess_input(review)
    sentiment = model.predict(processed_review)
    return "Positive" if sentiment>0.5 else "Negative"

# Streamlit App
st.title("IMDb Movie Reviews Sentiment Analysis")

# User Input
movie_review = st.text_input('Movie Review')

# Output
st.write(predict_sentiment(movie_review))