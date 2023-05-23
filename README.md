# Disaster Response - Machine Learning Pipeline and Web Development Project

## Introduction 
The goal of this project was to develop a machine learning model to classify messages that could be related to disasters, so some action could be taken accordingly. Also, another aim was to implement the tool as a web application.

## Instalation
The project includes scripts written in Python and HTML. Also, the dataset are available in csv and sql (.db) formats. In order to run the scripts in Python, the following libraries are needed: json, plotly, pandas, nltk,stem, flask, joblib, sqlalchemy, sklearn.

## File and Folder Descriptions

1. Data: This folder contains the raw datasets (categories.csv and messages.csv), the cleaned database (DisastersProject.db) and the script used to process the data.
2. app: This folder is where the web app runs. It has a folder templates with the html scrips to run the application. Also, the Flask App has a python script (run.py) that should be used to trigger the application.
3. models: This folder contains the final model (final_model.pkl) and the python script in which the model has been trained and exported.
4. screenshots: In this folder there are five screenshots of the web app. 
5. ETL Pipeline Preparation.ipynb: This is a Jupyter notebook showing the data preparation step by step using ETL framework.
6. ML Pipeline Preparation.ipynb: This is a jupyter notebook showing the Machine Learning Pipelation Preparation step by step.
7. etl_pipeline.py: This is a Python script that cleans the data and export it as a database to be further analyzed with Machine Learning.

## Instructions:
1. Download the repository to a folder in your local machine.
    <br>A. If you want to run the ETL and ML pipeline, simply run the scripts using your favorite Python IDE. Make sure to use the root folder as your working directory. Run first the ETL file, then ML. 
    <br>B. If you only want to use the Web App, simply run the file run.py and set the same folder as working directory. This script will make a local server. With the script running, open your browser and type http://0.0.0.0:3001/ or http://localhost:3001/

## Screen Shots
![Fig_1](/screenshots/head_main_page.jpg){: width="400px" height="200px" align="center"}
![Fig_2](/screenshots/Bar_plot_amount_by_genres.jpg){: width="400px" height="300px" align="center"}
![Fig_3](/screenshots/bar_plot_amount_of_messages_by_categories.jpg){: width="400px" height="300px" align="center"}
![FIg_4](/screenshots/histogram_with_message_length.jpg){: width="400px" height="300px" align="center"}
![Fig_5](/screenshots/classification_example.jpg){: width="400px" height="400px" align="center"}

## Disclaimer
The template used in this document was not developed by me. It is based on a pre-existing template created by [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025). I have made modifications and additions to suit my needs, but the original structure and are virtually the same.






