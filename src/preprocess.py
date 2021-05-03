import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from utils import load_config

config = load_config()

tz = pytz.timezone('Asia/Kolkata')

data = pd.read_csv('data/raw/Binance_BTCUSDT_minute.csv', parse_dates=['date'])
data = data.sort_values(by = 'date').reset_index(drop = True)
data.unix = data.unix / 1000


def new_feature(data):
    data['previous-close'] = data.shift(1)['close']
    data['close-change'] = data['close'] - data['previous-close']
    data['close-change'] = data['close-change'].fillna(0)
    return data

def process(data):
    df = pd.DataFrame()
    df['date'] = [datetime.fromtimestamp(i, tz) for i in data['unix']]
    df['day_of_month'] = [i.day for i in df['date']]
    df['day_of_week'] = [i.dayofweek for i in df['date']]
    df['week_of_year'] = [i.week for i in df['date']]
    df['month'] = [i.month for i in df['date']]
    df['open'] = [i for i in data['open']]
    df['high'] = [i for i in data['high']]
    df['low'] = [i for i in data['low']]
    df['close'] = [i for i in data['close']]
    df['close_change'] = [i for i in data['close-change']]
    df['date'] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in df['date']]
    return df

data = new_feature(data)
final_data = process(data)

train_size = int(final_data.shape[0] * config['train_ratio'])

train_data, test_data = final_data[:train_size], final_data[train_size+1:]

train_data.to_csv('data/feature/train.csv', index=False)
test_data.to_csv('data/feature/test.csv', index=False)