# test.py

import json
import pickle
import dataset
import logging
import nltk


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point of the script.
    """
    
    try:
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("Error: config.json file not found.")
        return
    except json.JSONDecodeError:
        logging.error("Error: config.json file is not a valid JSON file.")
        return

    # Load the trained model
    try:
        with open(config['save_model_path'], 'rb') as f:
            pipeline_model = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Model not found at {config['save_model_path']}")
        return

    # Get the SMS input
    sms_input = input("Enter an SMS: ")

    # Clean the SMS input
    cleaned_sms = dataset.clean_text(sms_input, apply_stemming=False, apply_lemmatization=False)
    logging.info(f"Cleaned SMS: {cleaned_sms}")

    # Classify the SMS input
    try:
        prediction = pipeline_model.predict([cleaned_sms])
        prediction_proba = pipeline_model.predict_proba([cleaned_sms])
        logging.info(f"Prediction: {prediction[0]}")
        logging.info(f"Prediction Probability: {prediction_proba[0][1]}")
    except Exception as e:
        logging.error(f"Error during classification - {str(e)}")
        return

if __name__ == "__main__":
    main()