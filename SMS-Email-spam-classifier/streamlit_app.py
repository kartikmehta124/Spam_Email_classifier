import streamlit as st 
import pickle
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]       
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load the pre-trained vectorizer
with open('C:/Users/maste/Downloads/Email Spam Classifier/SMS-Email-spam-classifier/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)


# Load the pre-trained model
with open('C:/Users/maste/Downloads/Email Spam Classifier/SMS-Email-spam-classifier/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# This step is important: Fit the vectorizer on some data (you can use your training data)
# This ensures that the vectorizer has the vocabulary information it needs
# If your training data is large, you may need to re-run the training script to fit the vectorizer
# and then save the vectorizer again.
# For now, I'm assuming you have a DataFrame df_train with 'transform_text' column.
df_train = pd.read_csv('C:/Users/maste/Downloads/Email Spam Classifier/SMS-Email-spam-classifier/spam.csv', encoding ='latin-1')
df_train['transform_text'] = df_train['v2'].apply(transform_text)
tfidf.fit(df_train['transform_text'])

st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From :-", ["Via Email ", "Via SMS", "other"])

if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')
