import pandas as pd
from tqdm.auto import tqdm
import yfinance as yf
import traceback

data = pd.read_csv('/home/apodtikhov2/download_data/labels/stock_tickers.csv')
ticker_list = []
for i, row in tqdm(data.iterrows(), total=data.shape[0]):
    ticker_name = row['Ticker']
    ticker_list.append(ticker_name)
    '''
    try:
        ticker = yf.Ticker(ticker_name)
        ticker.history(period='2y')[['Close']].to_csv('/home/apodtikhov2/download_data/series/' + ticker_name + '.csv')
    except Exception:
        traceback.print_exc()
    '''

 
 
# Here we use yf.download function
data = yf.download(
     
    # passes the ticker
    tickers=ticker_list[:10000],
    threads=True, # Set thread value to true
    period='max',
    # used for access data[ticker]
    group_by='ticker',
 
)
 
# used for making transpose
# data = data.H
print(data.head())
print(data.columns)
cols = {}
for c in data.columns:
    if c[1] == 'Adj Close':
        cols[c] = c[0]
data = data[cols.keys()]
data.columns = data.columns.get_level_values(0)
data.to_csv('stocks.csv')

data = pd.read_csv('/home/apodtikhov2/download_data/labels/stock_tickers.csv')
ticker_list = []
for i, row in tqdm(data.iterrows(), total=data.shape[0]):
    ticker_name = row['Ticker']
    ticker_list.append(ticker_name)
    '''
    try:
        ticker = yf.Ticker(ticker_name)
        ticker.history(period='2y')[['Close']].to_csv('/home/apodtikhov2/download_data/series/' + ticker_name + '.csv')
    except Exception:
        traceback.print_exc()
    '''

 
 
# Here we use yf.download function
data = yf.download(
     
    # passes the ticker
    tickers=ticker_list[10000:],
    threads=True, # Set thread value to true
    period='max',
    # used for access data[ticker]
    group_by='ticker',
 
)
 
# used for making transpose
# data = data.H
print(data.head())
print(data.columns)
cols = {}
for c in data.columns:
    if c[1] == 'Adj Close':
        cols[c] = c[0]
data = data[cols.keys()]
data.columns = data.columns.get_level_values(0)
data.to_csv('stocks1.csv')