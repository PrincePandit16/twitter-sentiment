import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# 1. Download necessary NLTK data
nltk.download('stopwords')

# 2. Load the Model and Vectorizer
# We use @st.cache_resource so it only loads once, making the app faster
@st.cache_resource
def load_model():
    model = pickle.load(open('trained_model.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

# 3. Define the Stemming Function (Must match your training code!)
def stemming(content):
    port_stem = PorterStemmer()
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # Stem and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                       if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

# 4. Streamlit UI
st.title('Twitter Sentiment Analysis 🐦')
st.write('Enter a tweet below to see if it is Positive or Negative.')

# Text Input
user_input = st.text_area("Enter Tweet:", height=150)

if st.button('Analyze Sentiment'):
    if user_input:
        # A. Preprocess the input
        cleaned_text = stemming(user_input)
        
        # B. Vectorize (Transform) the input
        # Note: We use .transform(), NOT .fit_transform()
        vectorized_input = vectorizer.transform([cleaned_text])
        
        # C. Predict
        prediction = model.predict(vectorized_input)
        
        # D. Display Result
        if prediction[0] == 0:
            st.error("Prediction: Negative Tweet 😠")
        else:
            st.success("Prediction: Positive Tweet 😃")
    else:
        st.warning("Please enter some text first.")