import os
import pandas as pd
import psycopg2 as ps

def get_csv_filenames() -> list:
    """Get names of csv file inside the current working directory
    
    Args: None
    
    Returns: 
        file_name (list): a list of csv file names in the current working directory
    """
    file_names = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".csv"):
            file_names.append(file)
    return file_names


def configure_folder_directory(csv_filenames: list, folder_name: str) -> None:
    """Create new folder if not exists and move csv files to the newly created folder
    
    Args: 
        csv_filenames (list): a list of csv filenames
        folder_name (str): name of the folder to hold files
        
    Raises: pass error in case folder already exists
    
    Returns: None
    """
    try: 
        mkdir = f'mkdir {folder_name}'
        os.system(mkdir)       # create new folder
    except:
        pass
    
    for name in csv_filenames:
        mv_file = f"mv '{name}' {folder_name}"
        try:
            os.system(mv_file)  # move files
        except:
            raise Exception('No csv file to move')
        

def create_df(folder_name: str, csv_filenames: list) -> dict:
    """Load dataset and create dataframe object for all files in folder
    
    Args:
        folder_name (str): name of the folder to dataset files
        csv_filenames (list): a list of csv filenames
        
    Returns:
        df (dict): a dictionary with file names as key and a dataframe as value
    """
    dir_path = os.getcwd() + '/' + folder_name + '/'  # get path of the folder containing csv files

    df = {}  # initiate empty dictionary to store df
    for file in csv_filenames:
        try:
            df[file] = pd.read_csv(dir_path + file)  # load data
        except UnicodeDecodeError:                   # to capture utf-8 encoding error
            df[file] = pd.read_csv(dir_path + file, encoding="ISO-8859-1")  
        print(f"{file} dataset is loaded")
    return df


def clean_tablename(filename: str) -> str:
    """Clean table name using csv filename
    
    Args:
        filename (str): filename to be cleaned
    
    Returns: 
        cleaned_name (str): cleaned filename
    """
    cleaned_name = filename.casefold().replace(" ", "").replace("-","_").replace(r"/","_")\
        .replace("\\","_").replace("$","").replace("%","")
    cleaned_name = cleaned_name.split('.')[0]       # store file name only, obmit file extension
    return str(cleaned_name)


def clean_colname(data: pd.DataFrame) -> tuple:
    """Clean column names in csv file

    Args:
        data (pd.DataFrame): dataframe containing columns to be cleaned

    Returns:
        (sql_col_str, cleaned_colname) (tuple): SQL column header with SQL data type and cleaned column name
    """
    cleaned_colname = [x.casefold().replace(" ", "_").replace("-","_").replace(r"/","_")\
        .replace("\\","_").replace(".","_").replace("$","").replace("%","") for x in data.columns] 
    #processing data
    replacements = {'timedelta64[ns]': 'varchar', 'object': 'varchar', 'float64': 'float',
        'int64': 'int', 'datetime64': 'timestamp'}
    
    sql_col_str = ", ".join(f"{name} {datatype}" for (name, datatype) in \
        zip(cleaned_colname, data.dtypes.replace(replacements)))  # SQL column header with SQL datatype
    
    return sql_col_str, cleaned_colname


def connect_to_db(host_name: str, dbname: str, username: str, password: str, port: str):
    """Connect to the databse

    Args:
        host_name (str): database host name
        dbname (str): database name
        username (str): username
        password (str): password
        port (str): port

    Raises:
        OperationalError: if program fails to connect to db
    """
    try:
        conn = ps.connect(host=host_name, database=dbname, user=username, password=password, port=port)
    except ps.OperationalError as e:
        raise e
    else:
        print('Successfully connected to DB!')
        return conn
    

def upload_to_db(cursor, tbl_name: str, col_str: str, file: str, dataframe: pd.DataFrame, column_names: list):
    """Create table, upload data to DB
    
    Args: 
        cursor: cursor connection to DB
        tbl_name (str): name of the table to be created in DB
        col_str (str): column name with corresponding SQL datatype
        file (str): name of csv file
        dataframe (pd.DataFrame): dataset
        column_names (list): name of columns in csv file (to save file)
    
    Returns: None
    """
    cursor.execute(f"DROP TABLE IF EXISTS {tbl_name};")      #drop table with same name
    cursor.execute(f"CREATE TABLE {tbl_name} ({col_str});")  #create table
    print(f'Table {tbl_name} created.') 

    dataframe.to_csv(file, header=column_names, index=False, encoding='utf-8')  #save df to csv
    my_file = open(file)  #open the csv file
    print(f'File {file} opened in memory')

    SQL_STATEMENT = f"""COPY {tbl_name} FROM STDIN WITH CSV HEADER DELIMITER AS ','"""      #upload to db

    cursor.copy_expert(sql=SQL_STATEMENT, file=my_file)
    print(f'File {file} copied to DB')
    print(f'Table {tbl_name} imported to DB.')
