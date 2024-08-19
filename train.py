# train.py

import argparse
import json
import dataset
import classification
import pandas as pd
import logging




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

    parser = argparse.ArgumentParser(description='SMS Spam Classification')
    parser.add_argument('--data-path', default=config['data_path'], help='Path to the dataset file')
    parser.add_argument('--model', choices=['logistic_regression', 'multinomial_nb'], default=config['model'],
                        help='Choose the classification model')
    parser.add_argument('--stem', action='store_true', help='Apply stemming')
    parser.add_argument('--lemmatize', action='store_true', help='Apply lemmatization')
    args = parser.parse_args()

    try:
        df = dataset.load_data(args.data_path)
    except FileNotFoundError:
        logging.error(f"Error: File not found at {args.data_path}")
        return
    except pd.errors.EmptyDataError:
        logging.error(f"Error: File at {args.data_path} is empty.")
        return
    except pd.errors.ParserError:
        logging.error(f"Error: Unable to parse file at {args.data_path}.")
        return

    try:
        X, y = dataset.prepare_data(df, apply_stemming=args.stem, apply_lemmatization=args.lemmatize)
    except KeyError:
        logging.error("Error: Invalid data format. Please check the data file.")
        return

    if args.model == 'logistic_regression':
        model = classification.LogisticRegressionModel()
    elif args.model == 'multinomial_nb':
        model = classification.MultinomialNBModel()
    else:
        raise ValueError('Invalid model choice')

    try:
        model.classify(X, y, config['save_model_path'])
    except Exception as e:
        logging.error(f"Error: An error occurred during classification - {str(e)}")
        return

if __name__ == "__main__":
    main()




## Example :
# python main.py --lemmatize
# python main.py
# python main.py --stem --model logistic_regression