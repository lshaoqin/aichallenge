import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import numpy as np

def add_desclength_col(df):
    """
    Add a column for the length of the description
    """
    df['DescriptionLength'] = df['Description'].str.split().str.len().fillna(0)
    df['DescriptionLengthBins'] = pd.cut(df['DescriptionLength'], bins=[-1, 0, 22, 45, 82, np.inf], labels=['0', '1-21', '22-44', '45-81', '81+'])
    return df

def split_text_categorical(df):
    """
    Split the columns into text and categorical columns, 
    dropping unwanted columms
    """
    y = df['AdoptionSpeed']

    # Column 32 onwards are all derived values or labels which have already been encoded
    # We will keep BreedPure, ColorAmt and NameorNO though because the information may be lost when we remove the original columns
    x = df.iloc[:,:34].drop('AdoptionSpeed', axis=1)
    x = x.drop(['Name', 'RescuerID', 'PetID'], axis=1)

    # Remove numerical columns because they have been binned
    x = x.drop(['Breed1', 'Breed2', 'Age', 'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt'], axis=1)

    x_text = df['Description']
    x_cat = x.drop('Description', axis=1)

    return x_text, x_cat, y

def split_train_test(x, y, test_size = 0.2):
    """
    Split the data into train and test sets
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
    return x_train, x_test, y_train, y_test

def split_train_test_val(x, y, test_size = 0.2, val_size = 0.2):
    """
    Split the data into train, test and validation sets
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = val_size)

    return x_train, x_test, x_val, y_train, y_test, y_val

def ordinal_encode(x_train, x_test, x_val = None):
    """
    Ordinal encode the categorical columns
    """
    x_train_ord = x_train.copy()
    oe = OrdinalEncoder()
    x_train_ord = oe.fit_transform(x_train_ord)

    x_test_ord = x_test.copy()
    x_test_ord = oe.transform(x_test_ord)

    if x_val is not None:
        x_val_ord = x_val.copy()
        x_val_ord = oe.transform(x_val_ord)
        return x_train_ord, x_test_ord, x_val_ord
    else:
        return x_train_ord, x_test_ord

def one_hot_encode(x_train, x_test, x_val = None):
    """
    One hot encode the categorical columns
    """
    x_train_hot = x_train.copy()
    ohe = OneHotEncoder()
    x_train_hot = ohe.fit_transform(x_train_hot)

    x_test_hot = x_test.copy()
    x_test_hot = ohe.transform(x_test_hot)

    if x_val is not None:
        x_val_hot = x_val.copy()
        x_val_hot = ohe.transform(x_val_hot)
        return x_train_hot, x_test_hot, x_val_hot
    else:
        return x_train_hot, x_test_hot
    
def preprocess_text(df):
    """
    Preprocess the text data
    """
    def clean_text(text):
        text = text.str.lower()
        text = text.str.replace(r'[^a-z\s0-9]', '')
        text = text.str.replace(r'\s+', ' ')
        return text
    
    df['Description'] = clean_text(df['Description'])

    return df

def correct_spellings(df): # Slow!
    """
    Correct spelling errors in descriptions
    """
    def correct_spelling(text):
        if pd.isnull(text):
            return text
        tb = TextBlob(text)
        return str(tb.correct())
    
    df['Description'] = df['Description'].apply(correct_spelling)
    return df







