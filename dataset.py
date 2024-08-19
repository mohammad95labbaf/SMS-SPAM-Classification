# dataset.py

import pandas as pd
from nltk.corpus import stopwords
import re
import logging
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk



STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def load_data(data_path):
    """
    Load data from a CSV file.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded data.
    """

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {data_path} - {str(e)}")
        raise Exception(f"Error loading data from {data_path} - {str(e)}")

def prepare_data(df, apply_stemming=False, apply_lemmatization=False):

    """
    Prepare data for classification.

    Args:
        df (pd.DataFrame): Loaded data.
        apply_stemming (bool, optional): Apply stemming. Defaults to False.
        apply_lemmatization (bool, optional): Apply lemmatization. Defaults to False.

    Returns:
        tuple: Prepared data (X, y).
    """
    
    try:
        df['clean_text'] = df['messages'].apply(lambda x: clean_text(x, apply_stemming, apply_lemmatization))
        X = df['clean_text']
        y = df['label']
        logging.info("Data preparation completed")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing data - {str(e)}")
        raise Exception(f"Error preparing data - {str(e)}")

def clean_text(text, apply_stemming=False, apply_lemmatization=False):

    """
    Clean text data.

    Args:
        text (str): Text data.
        apply_stemming (bool, optional): Apply stemming. Defaults to False.
        apply_lemmatization (bool, optional): Apply lemmatization. Defaults to False.

    Returns:
        str: Cleaned text data.
    """

    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stopwords
        text = " ".join(word for word in text.split() if word not in STOPWORDS)
        
        if apply_stemming:
            # Apply stemming
            text = " ".join(stemmer.stem(word) for word in text.split())
        
        if apply_lemmatization:
            # Apply lemmatization
            text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
        
        # Normalize text (optional)
        text = text.strip()
        
        logging.debug(f"Cleared text: {text}")
        return text
    except Exception as e:
        logging.error(f"Error cleaning text - {str(e)}")
        raise Exception(f"Error cleaning text - {str(e)}")