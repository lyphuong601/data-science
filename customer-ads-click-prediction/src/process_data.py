import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme()

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:,.3f}'.format)

##########################################################
# Load - Data Overview Routines
##########################################################

def load_data(filename: str, **kwargs) -> pd.DataFrame:
    """Read data from a filename and output it as a dataframe"""
    df = pd.read_csv(filename, **kwargs)
    return df


def data_overview(data: pd.DataFrame) -> str:
    print('1. Data description: ')
    print(data.describe(include='all'))  # Description of dataset
    print(f'2. Shape of dataset: {data.shape}')
    print('3. Columns datatype: ')
    for group, column in data.columns.to_series().groupby(data.dtypes):  # Datatype of each column
        print(group, end='\t| ')
        for name in column:
            print(name, end=', ')
        print()


def check_missing_data(data: pd.DataFrame) -> pd.Series:
    """Check for missing data in the df (display in descending order)"""
    result = ((data.isnull().sum() * 100)/ len(data)).sort_values(ascending=False)
    return result


def zscore_normalize_features(X):
    mu     = np.mean(X, axis=0)  
    sigma  = np.std(X, axis=0) 
    X_norm = (X - mu)/sigma      
    return (X_norm, mu, sigma)
