import datetime
import pandas as pd
from pandas import read_csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  PRIIPS_calc.PRIIPS_calc import PRIIPS_stats_df, PRIIPS_stats_bootstrap, plot_bootstrap_priips
sns.set_theme() # use the sns theme for plotting - just more attractive!

ANNUALISE = 256 # Number of trading days in the year for annualising = 52*5 Actual number of days often less
# For example, PRIIPS uses 256

# Total Period
MAX_DATE_FILTER = datetime.datetime(2023, 10, 27) # end of sample 'March 2022'
MIN_DATE_FILTER = datetime.datetime(2004, 1, 1) # beginning of sample 'Jan 2004' - Jan 7 results in similar


date_ranges = {
                '2004-2010' : (datetime.datetime(2004, 1, 1), datetime.datetime(2010, 12, 31)),
                '2010-2016' : (datetime.datetime(2011, 1, 1), datetime.datetime(2016, 12, 31)),
                '2017-2023' : (datetime.datetime(2017, 1, 1), datetime.datetime(2023, 10, 27))
                }
# Read in the price history
df_price_history = read_csv('https://raw.githubusercontent.com/TimWilding/FinanceDataSet/main/PRIIPS_Data_Set.csv', parse_dates=[1])


# Build the Geometric returns
df_price_history['LogPrice'] = np.log(df_price_history['Close'])
df_price_history.sort_values(['Index', 'Date'], inplace=True)
df_price_history['LogReturn'] = 100*df_price_history.groupby('Index')['LogPrice'].diff()
df_price_history = df_price_history[df_price_history.Date>=MIN_DATE_FILTER] # Note - this removes NaNs from initial price points in LogReturn column


dct_df_samples = {}

for key, value in date_ranges.items():
    df_sample = df_price_history[(df_price_history.Date>=value[0]) & (df_price_history.Date<=value[1])]
    dct_df_samples[key] = df_sample

df_sample_prices = df_price_history[(df_price_history.Date>MIN_DATE_FILTER) & (df_price_history.Date<=MAX_DATE_FILTER)]
pivoted_df = df_sample_prices.pivot(index='Date', columns='Index', values='LogReturn')
Y = pivoted_df.values
mask = np.logical_not(np.any(np.isnan(Y), axis=1))
Z = Y[mask,:] # only use columns of Y that don't contain any NaNs


def plot_index_prices(df, col_name, axis_label):
    """
    Plot each of the indices on chart for the performance dataset
    """
    df_sort = df.sort_values(['Index', 'Date'])
    df_sort['CumLogRet'] = df.groupby('Index')[col_name].cumsum()
    df_sort['IndexedPrice'] = df_sort.groupby('Index')['CumLogRet'].transform(lambda x: 100*np.exp((x -x.iloc[0] + 1.0)/100.0))

# Plot the time series for each asset using Seaborn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df_sort, x='Date', y='IndexedPrice', hue='Index')
    plt.ylabel(axis_label)

# Display the plot
    plt.show()


def plot_stats_col(df, col_name, axis_label, chart_title):
    """
    Plot each of the indices on chart from the statistics daata set
    """
    df_sort = df.sort_values(['Identifier', 'Date'])

# Plot the time series for each asset using Seaborn
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_sort, x='Date', y=col_name, hue='Identifier')
    plt.ylabel(axis_label)
    plt.title(chart_title)

# Display the plot
    plt.show()

def build_asset_stats(df_price_sample, period_desc):
    """
    Build a small dataframe containing the asset statistics from df_price_sample
      - returns min & max of time period, mean percent return from price, std percent return from price, mean total pct return
    """
    df_agg = df_price_sample.groupby('Index').agg({
                                        'Date' : ['min', 'max', 'count'],
                                        'LogReturn' : ['mean', 'std']
                                        })
    df_agg[('LogReturn', 'mean')] = ANNUALISE*df_agg[('LogReturn', 'mean')]
    df_agg[('LogReturn', 'std')] = np.sqrt(ANNUALISE)*df_agg[('LogReturn', 'std')]
    df_agg['Period'] = period_desc
    return df_agg

df_stats_history = read_csv('https://raw.githubusercontent.com/TimWilding/FinanceDataSet/main/PRIIPS_Stats.csv', encoding='utf-8', parse_dates=[1], skiprows=3)
df_stats_history = df_stats_history[df_stats_history.Date>=MIN_DATE_FILTER] # Note - this removes NaNs from initial price points in LogReturn column

df_sample_10y = df_price_history[(df_price_history.Date>datetime.datetime(2013, 10, 27)) & (df_price_history.Date<=datetime.datetime(2023, 10, 27))]
df_bs = PRIIPS_stats_df(df_sample_10y)
print(plot_bootstrap_priips(df_bs))