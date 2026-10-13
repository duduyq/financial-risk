#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np
import time
import urllib3
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

proxies = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897',
}

SYMBOL = 'BTCUSDT'
INTERVAL = '5m'
DAYS = 180
BATCH_SIZE = 1000

end_time = int(datetime.now().timestamp() * 1000)
start_time = int((datetime.now() - timedelta(days=DAYS)).timestamp() * 1000)

print("=" * 60)
print(f"币安 {SYMBOL} {INTERVAL} K线数据下载工具")
print(f"时间范围: 近 {DAYS} 天")
print("=" * 60)

all_rows = []
cursor = start_time
req_count = 0

while cursor < end_time:
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'startTime': cursor,
        'endTime': end_time,
        'limit': BATCH_SIZE,
    }

    try:
        response = requests.get(url, params=params, proxies=proxies, timeout=30, verify=False)
        response.raise_for_status()
        req_count += 1
        data = response.json()
    except Exception as e:
        print(f"请求失败 (第 {req_count + 1} 次): {e}")
        break

    if not data:
        break

    all_rows.extend(data)
    last_open_time = data[-1][0]
    cursor = last_open_time + 5 * 60 * 1000

    if req_count % 10 == 0:
        dt = pd.to_datetime(last_open_time, unit='ms', utc=True)
        print(f"  已请求 {req_count} 次, 累计 {len(all_rows)} 条, 当前至 {dt}")

    time.sleep(0.15)

print(f"\n请求完成: 共 {req_count} 次请求, {len(all_rows)} 条原始数据")

if not all_rows:
    print("未获取到数据，请检查网络和代理设置")
    exit(1)

COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore']

df = pd.DataFrame(all_rows, columns=COLS)
df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')

for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype(int)

df = df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)

df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna(subset=['log_return']).reset_index(drop=True)

output_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'log_return']
df_out = df[output_cols].copy()

output_file = '/Users/kevinma/Downloads/中国社会科学院大学/金融风险管理/小组汇报/btc_5min_6months.csv'
df_out.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n数据预览:")
print(df_out.head(10))
print(f"\n数据概览:")
print(f"  总行数:     {len(df_out):,}")
print(f"  时间范围:   {df_out['datetime'].iloc[0]} ~ {df_out['datetime'].iloc[-1]}")
print(f"  价格区间:   ${df_out['close'].min():,.2f} ~ ${df_out['close'].max():,.2f}")
print(f"  收益率均值: {df_out['log_return'].mean():.6f}")
print(f"  收益率标准差: {df_out['log_return'].std():.6f}")
print(f"\n已保存至: {output_file}")
