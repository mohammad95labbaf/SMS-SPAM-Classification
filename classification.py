# classification.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import logging
import pickle

class ClassificationModel:
    def __init__(self, model):
        self.model = model
        """
        Initialize a classification model.

        Args:
            model: Classification model.
        """

    def classify(self, X, y, save_model_path):
        
        """
        Classify data using the model.

        Args:
            X: Feature data.
            y: Target data.
        """

        try:
            # train test split
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
            logging.info("Data split completed")
            # model training
            pipeline_model = Pipeline([('vect', CountVectorizer()),
                                      ('tfidf', TfidfTransformer()),
                                      ('clf', self.model)])
            pipeline_model.fit(x_train, y_train)
            logging.info("Model training completed")

            # model Testing
            y_pred = pipeline_model.predict(x_test)
            y_pred_proba = pipeline_model.predict_proba(x_test)

            logging.info(f"Accuracy: {pipeline_model.score(x_test, y_test)*100}")
            logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba[:, 1])*100}")
            logging.info("Model testing completed")

            print(classification_report(y_test, y_pred))
            # Save the trained model
            with open(save_model_path, 'wb') as f:
                pickle.dump(pipeline_model, f)
            logging.info(f"Model saved at {save_model_path}")

        except Exception as e:
            logging.error(f"Error during classification - {str(e)}")
            raise Exception(f"Error during classification - {str(e)}")
        


class LogisticRegressionModel(ClassificationModel):
    def __init__(self):
        super().__init__(LogisticRegression())

class MultinomialNBModel(ClassificationModel):
    def __init__(self):
        super().__init__(MultinomialNB())