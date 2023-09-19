import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report, accuracy_score

from lightgbm import LGBMClassifier

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_cleaned', engine)
    X = df.message
    Y = df[df.columns[4:]] 

    category_names =  df[df.columns[4:]].columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes a given text into words.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list of str: A list of tokens (words) from the input text.
    """
    tokens = word_tokenize(text)
    return tokens


def build_model(X_train,Y_train):

    subset_size = 100#round(0.2*len(X_train))  # size of the random subset for parameter tuning

    # sample a subset from training data
    random_indices = np.random.choice(len(X_train), size=subset_size, replace=False)
    X_train_subset = X_train.iloc[random_indices]
    y_train_subset = Y_train.iloc[random_indices]

    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),  # Use n-grams here (1, 2) for bigrams
    ('clf', MultiOutputClassifier(LGBMClassifier()))])

    parameters = {
        'clf__estimator__learning_rate': [0.01, 0.1],#, 0.2, 0.3],      # Learning rate
        'clf__estimator__n_estimators': [100],#, 200, 300, 400],         # Number of boosting rounds
        'clf__estimator__subsample': [0.7, 0.8],#, 0.9, 1.0],           # Fraction of data used for training
        'clf__estimator__colsample_bytree': [0.7]#, 0.8, 0.9, 1.0],    # Fraction of features used for each tree
       }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train_subset, y_train_subset)

    best_params = cv.best_params_
    best_model = pipeline.set_params(**best_params)

    return best_model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    print(y_pred)
    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)
    print(y_pred_df.shape)
    print(Y_test.shape)

    for i, column_name in enumerate(Y_test.columns):  
        y_true_column = Y_test[column_name]
        y_pred_column = y_pred[:, i]

        # Calculate and print classification report for each column
        accuracy = accuracy_score(y_true_column, y_pred_column)

        report = classification_report(y_true_column, y_pred_column)
        print(f"Classification Report for '{column_name}':")
        print(report)
        print(f"Accuracy: {accuracy}")


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data.../n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model( X_train, Y_train)
        
        print('Training model...')
        subset_size = round(len(X_train)*0.5)
        random_indices = np.random.choice(len(X_train), size=subset_size, replace=False)
        X_train_subset = X_train.iloc[random_indices]
        Y_train_subset = Y_train.iloc[random_indices]
        model.fit(X_train_subset, Y_train_subset)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


