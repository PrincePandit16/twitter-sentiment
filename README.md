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
```


## 🚀How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/PrincePandit16/twitter-sentiment.git](https://github.com/PrincePandit16/twitter-sentiment.git)
cd twitter-sentiment
```
### 2. Install Dependencies
Make sure you have Python installed. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3.Run the app
```bash
streamlit run app.py
```
## 🧠 Model Details

### Preprocessing Pipeline
Before the text is fed into the model, the following steps are applied:
* **Regex Cleaning:** Removes non-alphabetic characters (punctuations, numbers, emojis).
* **Lowercasing:** Converts all text to lowercase for consistency.
* **Porter Stemming:** Reduces words to their root form (e.g., "running" -> "run").
* **Stopword Removal:** Removes common words (like "is", "the", "and") that don't carry sentiment.

### Feature Extraction
* **TF-IDF Vectorizer:** Converts text into numerical vectors (maximum 500,000 features).

### Algorithm
* **Logistic Regression:** Chosen for its efficiency and strong performance on binary text classification tasks.
* **Accuracy:** ~77.6% on Test Data.



## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

## 👤 Author
**PrincePandit16**
* GitHub: [@PrincePandit16](https://github.com/PrincePandit16)
