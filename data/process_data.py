
import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loading and transforming the data to get it into a format on which a model can be estimated.

    Args:
        messages_filepath (str): path of disaster_messages.csv
        categories_filepath (str): path of disaster_categories

    Returns:
        DataFrame: DataFrame containing the messages and the associated categories 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge messages and categories
    df = messages.merge(categories, how= 'inner', on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =  row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis =1)
    return df

def clean_data(df):
    """Remove duplicates and nonsense expressions to clean the data

    Args:
        df (DataFrame): Initial data containing messages and their categories

    Returns:
        DataFrame: cleaned DataFrame
    """
    df = df.drop_duplicates()
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filename):
    """save the data to a SQL Database

    Args:
        df (DataFrame): data to be saved
        database_filename (str): filename of the database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_cleaned', engine, index=False, if_exists='replace'  )


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
     main()
