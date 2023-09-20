# Disaster Response Pipeline Project

### Summary
This project is for the classification of disaster messages. The goal is to use data science and machine learning techniques to facilitate the efficient categorization of messages sent during disaster events. Categorizing the messages will allow them to be routed to the appropriate disaster relief organizations.

### Data Source 
Real-world disaster message data obtained from Appen (formerly Figure 8).

### Dependencies
- Python 3.x
- Pandas
- Scikit-Learn
- NLTK
- Flask
- Matplotlib (for data visualization)
- And more (refer to requirements.txt)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/trained_model2.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Acknowledgments
Special thanks to Appen (formerly Figure 8) for providing the disaster message dataset.
