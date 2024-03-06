import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from symspellpy import SymSpell
import numpy as np
import os
import urllib
import pkg_resources

def read_df(file_path):
    """
    Read a dataframe from a file
    """
    df = pd.read_csv(file_path)
    return df

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
    return x_train, x_test, y_train, y_test

def split_train_test_val(x, y, test_size = 0.2, val_size = 0.2):
    """
    Split the data into train, test and validation sets
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = val_size, random_state = 42)

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
    
def preprocess_text(series):
    """
    Preprocess the text data
    """
    def clean_text(text):
        text = str(text).lower()
        text = text.replace(r'[^a-z\s0-9]', '')
        text = text.replace(r'\s+', ' ')
        return text
    
    series = series.apply(clean_text)
    
    return series

def correct_spellings(series):
    """
    Correct spelling errors in descriptions
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # Check if dictionary exists
    if not os.path.isfile('frequency_dictionary_en_82_765.txt'):
        url = 'https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt'
        filename = 'frequency_dictionary_en_82_765.txt'
        urllib.request.urlretrieve(url, filename)
    
    if not os.path.isfile('frequency_bigramdictionary_en_243_342.txt'):
        url = 'https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_bigramdictionary_en_243_342.txt'
        filename = 'frequency_bigramdictionary_en_243_342.txt'
        urllib.request.urlretrieve(url, filename)

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
    )

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    def correct_spelling(text):
        if pd.isnull(text):
            return text
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term
        
    series = series.apply(correct_spelling)
    return series







