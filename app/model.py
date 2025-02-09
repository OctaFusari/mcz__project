import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import app.config as conf

import re
import nltk
import string
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scaricare le risorse necessarie
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lista di parole poco informative (da personalizzare in base al dataset)
low_info_words = {'government', 'president', 'party', 'country', 'state'}

# Funzione di pre-processing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Rimozione di link, menzioni e numeri
    text = re.sub(r'http\S+|www\S+|@\S+|\d+', '', text)
    
    # Rimozione di punteggiatura e conversione in minuscolo
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Tokenizzazione e lemmatizzazione
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in low_info_words]
    
    return text

def train_model(type__model):
    print("entrato in addestramento")

    if(type__model == "low"):
        model = conf.MODEL_LOW_PATH

    elif(type__model == "medium"):
        model = conf.MODEL_MEDIUM_PATH

    else:
        model = conf.MODEL_HIGH_PATH
    # Load dataset
    df = pd.read_csv(conf.DATASET_PATH)

    # Preprocess data
    df = df.dropna(subset=['Party', 'Tweet'])
    X = df['Tweet']
    y = df['Party']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("entrato a metà")
    if(type__model == "low"):

        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(n_estimators=70, random_state=42))
        ])

    elif(type__model == "medium"):
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        

    else:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, preprocessor=preprocess_text, ngram_range=(1,2))),  # Min_df per filtrare parole rare
            ('feature_selection', SelectKBest(chi2, k=500)),  # Selezione delle migliori feature
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save model
    with open(model, 'wb') as f:
        pickle.dump(pipeline, f)

def load_model(type__model):

    if(type__model == "low"):
        model = conf.MODEL_LOW_PATH

    elif(type__model == "medium"):
        model = conf.MODEL_MEDIUM_PATH

    else:
        model = conf.MODEL_HIGH_PATH

    with open(model, 'rb') as f:
        return pickle.load(f)