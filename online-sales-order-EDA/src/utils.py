"""
This is the code to draw graphs
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:,.3f}'.format)


def load_data(filename: str, **kwargs) -> pd.DataFrame:
    """Read data from a filename and output it as a dataframe"""
    df = pd.read_excel(filename, **kwargs)
    print(df.head())
    return df


def data_overview(df):
    print('1. Data description: ')
    print(df.describe(include='all'))  # Description of dataset
    print(f'2. Shape of dataset: {df.shape}')
    print('3. Columns datatype: ')
    for group, column in df.columns.to_series().groupby(df.dtypes):  # Datatype of each column
        print(group, end='\t| ')
        for name in column:
            print(name, end=', ')
        print()


def plot_bins(data: pd.Series, no_bin: int, group_name: list) -> pd.Series:
    """Cut data into different bins and assign names for each bin"""
    bins = np.linspace(data.min(), data.max(), no_bin)
    data = pd.cut(data, bins, labels=group_name, include_lowest=True)
    plt.bar(group_name, data.value_counts())
    plt.title("Selling Price Histogram")
    plt.xlabel("Selling price")
    plt.ylabel("Frequency")
    plt.show()


def check_missing_data(data: pd.DataFrame) -> pd.Series:
    """Check for missing data in the df (display in descending order)"""
    result = ((data.isnull().sum() * 100)/ len(data)).sort_values(ascending=False)
    return result


def corr_heatmap(data: pd.DataFrame):
    """Draw heatmap to show data correlation"""
    corr = data.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    palette = sns.diverging_palette(20, 220, n=256) # "YlGnBu"
    # Heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(corr, vmax=.5, mask=mask, annot=True, fmt='.2f', linewidths=.2, 
                cmap=palette)  # .get_figure().savefig('chart.png')
    plt.show()


def data_residual_plot(df, x_features, yname):
    fig,ax=plt.subplots(1,len(x_features),figsize=(12,4),sharey=True)
    for i in range(len(ax)):
        sns.residplot(x=x_features[i], y=yname, data=df, ax=ax[i], label=False)
    fig.suptitle("Residuals plot")
    plt.show()
  

def distribution_plot(actual_data, predicted_data):
    plt.figure(figsize=(8, 4))
    ax1 = sns.kdeplot(actual_data, color="r", label="Actual", clip=(-5, 70))
    ax2 = sns.kdeplot(predicted_data, color="b", label="Predicted", ax=ax1, clip=(-5, 70))
    plt.title("Actual vs Predicted values for order value")
    plt.xlabel('Order value')
    plt.ylabel('Proportion of orders')
    plt.legend(); plt.show(); plt.close()
    

def data_scatter_plot(x, x_features, y_actual, y_predicted):
    fig,ax=plt.subplots(1,len(x_features),figsize=(12,4),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x[x_features[i]], y_actual, label = 'Actual')
        ax[i].set_xlabel(x_features[i])
        ax[i].scatter(x[x_features[i]], y_predicted, label = 'Predicted')
    ax[0].set_ylabel("Order value"); ax[3].legend()
    fig.suptitle("Target vs prediction value")
    plt.show()
    

