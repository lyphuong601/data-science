import pandas as pd

def drop_na_data(data: pd.DataFrame, column_names: list) -> None:
    """Fill out null data with appropriate value, default value as `NA`"""
    for column in column_names:
        data.dropna(subset=[column],axis=0,inplace=True)
        

def change_dtypes(data: pd.DataFrame, column_names: list, to_type='float') -> pd.DataFrame:
    """Change data type of columns, default type is set to `float`"""
    data[column_names] = data[column_names].astype(to_type)
