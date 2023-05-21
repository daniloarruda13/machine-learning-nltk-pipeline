# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


def import_data():
    """
    Import data from a SQLite database into a pandas DataFrame and return X and Y.

    Returns:
        pandas.DataFrame: The imported data.
    """
    # create engine to connect to the database
    engine = create_engine('sqlite:///DisastersProject.db')

    # read the data from the table 'Messages_Categories' into a pandas DataFrame
    df = pd.read_sql_table('Messages_Categories', engine)

    # separate the input features from the labels
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])

    # return the input features and labels
    return X, y

def tokenize(text):
    """
    Tokenizes the input text into individual words, lemmatizes them, and returns 
    a list of cleaned tokens.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list: A list of cleaned tokens.
    """
    # tokenize the text (text should be cleaned already)
    tokens = word_tokenize(text)

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()

    #cleaning tokens: lowercase, remove whitespaces, lemmatize
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens



def build_model(pipeline_num):
    """
    Builds a machine learning model pipeline based on the provided pipeline number.

    Args:
        pipeline_num (int): An integer representing the pipeline number to be built.

    Returns:
        pipeline (Pipeline): A machine learning model pipeline object that has been 
        built based on the provided pipeline number.
    """
    if pipeline_num == 1:
        pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        pipeline = hyperparameter_tuning(pipeline)
        return pipeline

    else:
        pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
        ])
        return pipeline




def hyperparameter_tuning(pipeline):
    """
    This function performs hyperparameter tuning on a given pipeline using GridSearchCV.
    
    :param pipeline: A pipeline object that includes feature extraction and a classifier.
    :type pipeline: sklearn.pipeline.Pipeline
    
    :return: A GridSearchCV object that includes the best hyperparameters found during tuning.
    :rtype: sklearn.model_selection.GridSearchCV
    """
    parameters = {
    'clf__estimator__n_estimators': [100, 200, 300],
    'clf__estimator__max_depth': [ 5, 10]
    }
    cv = GridSearchCV(pipeline, parameters, cv=5)
    return cv



def metrics_figure(y_pred,y_test):
    """
	Calculates precision, recall, and f1-score for each category in y_test
	and plots a horizontal bar chart comparing the metrics. The function
	takes two parameters, y_pred and y_test, both of which are DataFrames.
	Returns nothing, but displays the plot. 
	"""

    #transforming to DF for use in the for loop
    data_frame = pd.DataFrame(y_pred, columns=y_test.columns)

    # Initialize empty lists to store metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Loop through columns and accumulate metrics
    for col_name, col_data in data_frame.items():
        report_dict = classification_report(y_test.loc[:, col_name], col_data, output_dict=True,zero_division=0)
        precision_scores.append(report_dict['weighted avg']['precision'])
        recall_scores.append(report_dict['weighted avg']['recall'])
        f1_scores.append(report_dict['weighted avg']['f1-score'])

    # Calculate average scores
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    # Create y-axis labels
    categories = data_frame.columns

    # Plotting the results
    y = np.arange(len(categories))
    height = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.barh(y - height, precision_scores, height, label='Precision', color='r')
    rects2 = ax.barh(y, recall_scores, height, label='Recall', color='g')
    rects3 = ax.barh(y + height, f1_scores, height, label='F1-Score', color='b')

    # Add average score vertical lines
    ax.axvline(avg_precision, color='r', linestyle='--', label='Avg Precision')
    ax.axvline(avg_recall, color='g', linestyle='--', label='Avg Recall')
    ax.axvline(avg_f1, color='b', linestyle='--', label='Avg F1-Score')

    # Add labels, title, and legend
    ax.set_xlabel('Scores')
    ax.set_ylabel('Categories')
    ax.set_title('Classification Metrics Comparison')
    ax.set_yticks(y)
    ax.set_yticklabels(categories)

    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add note with average values
    note = f'Avg Precision: {avg_precision:.2f}\nAvg Recall: {avg_recall:.2f}\nAvg F1-Score: {avg_f1:.2f}'
    ax.text(1.05, 0.5, note, transform=ax.transAxes, ha='left', va='center')

    # Display the plot
    plt.tight_layout()
    plt.show()



def export_model(pipeline):
    """
    Export the model to a pickle file.

    Args:
        pipeline (object): The trained pipeline object to be exported.

    Returns:
        None
    """
    # Export the model to a pickle file
    filename = 'final_model.pkl'
    pickle.dump(pipeline, open(filename, 'wb'))


def main():
    """
    Execute the main function of the script. This function imports data
    using the `import_data()` function, splits it into train and test sets
    using `train_test_split()`, builds two models using `build_model()`
    function with different parameters, fits the models using the training
    data, predicts the target variables using the test data, plots the
    metrics using `metrics_figure()`, and exports the second model using
    `export_model()`.

    Parameters:
    None

    Returns:
    None
    """
    X, y = import_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = build_model(1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_figure(y_pred, y_test)

    model_2 = build_model(2)
    model_2.fit(X_train, y_train)
    y_pred_2 = model.predict(X_test)
    metrics_figure(y_pred_2, y_test)
    
    export_model(model_2)

main()