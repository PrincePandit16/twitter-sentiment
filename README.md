# 🐦 Twitter Sentiment Analysis App

A Machine Learning web application that predicts whether a given tweet is **Positive** or **Negative**. Built with Python, Scikit-Learn, and Streamlit.

## 📋 About The Project
This project uses Natural Language Processing (NLP) to analyze the sentiment of text data. It was trained on the **Sentiment140** dataset, which contains 1.6 million tweets extracted using the Twitter API.

The application takes a user's input (a tweet), pre-processes the text (cleaning, stemming, stopword removal), and runs it through a trained Logistic Regression model to classify the sentiment.

## 📊 Dataset
The model was trained on the **Sentiment140 dataset from Kaggle**.
* **Source:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **Data Size:** 1.6 million tweets
* **Labels:** 0 (Negative), 4 (Positive) -> Mapped to 0 and 1.

## 🛠️ Technologies Used
* **Python** (Programming Language)
* **Streamlit** (Web Framework)
* **Scikit-Learn** (Machine Learning Library)
* **Pandas & NumPy** (Data Manipulation)
* **NLTK** (Natural Language Toolkit for Stopwords/Stemming)
* **Pickle** (Model Serialization)

## 📂 Project Structure
```text
twitter-sentiment-app/
│
├── app.py               # The main Streamlit web application
├── requirements.txt     # List of dependencies
├── trained_model.sav    # The trained Logistic Regression model
├── vectorizer.sav       # The fitted TF-IDF Vectorizer
├── Twitter.ipynb        # (Optional) The Jupyter Notebook used for training
└── README.md            # Project documentation
