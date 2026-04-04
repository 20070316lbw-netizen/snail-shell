import akshare as ak
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "QQ_Quant_DB/quant_lab.duckdb"

def get_latest_date(con):
    result = con.execute("SELECT MAX(date) FROM features_cn").fetchone()
    return result[0] if result[0] else None

def fetch_stock_data(ticker, start_date, end_date, retries=3):
    """从 akshare 获取股票历史数据，带重试机制"""
    symbol = ticker.split('.')[0]
    for i in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                    start_date=start_date.replace('-', ''), 
                                    end_date=end_date.replace('-', ''), 
                                    adjust="hfq")
            if df is not None and not df.empty:
                # 重命名列以匹配 prices_cn
                df = df.rename(columns={
                    '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
                    '收盘': 'close', '成交量': 'volume', '换手率': 'turn'
                })
                df['ticker'] = ticker
                df['date'] = pd.to_datetime(df['date']).dt.date
                return df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'turn']]
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Attempt {i+1} failed for {ticker}: {e}")
            time.sleep(2)
    return None

def calculate_features(df):
    """计算因子"""
    df = df.sort_values('date')
    
    # 收益率
    df['ret'] = df['close'].pct_change()
    
    # mom_20d: 20日收益率
    df['mom_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # mom_60d: 60日收益率
    df['mom_60d'] = df['close'] / df['close'].shift(60) - 1
    
    # mom_12m_minus_1m: 过去12个月排除最近1个月的动量
    # 约 252天 - 21天 = 231天左右的区间
    df['mom_12m_minus_1m'] = df['close'].shift(21) / df['close'].shift(252) - 1
    
    # vol_60d_res: 60日收益率标准差
    df['vol_60d_res'] = df['ret'].rolling(window=60).std()
    
    # sp_ratio: 20日夏普比率 (均值/标准差)
    df['sp_ratio'] = df['ret'].rolling(window=20).mean() / (df['ret'].rolling(window=20).std() + 1e-8)
    
    # turn_20d: 20日平均换手率
    df['turn_20d'] = df['turn'].rolling(window=20).mean()
    
    return df

def sync():
    con = duckdb.connect(DB_PATH)
    
    latest_date = get_latest_date(con)
    if not latest_date:
        logger.warning("Database empty, please initialize first.")
        return
    
    today = datetime.now().date()
    if latest_date >= today:
        logger.info("Database is already up to date.")
        # return # 强制刷新时可注释
    
    start_fetch = (latest_date + timedelta(days=1))
    end_fetch = today
    
    logger.info(f"Syncing from {start_fetch} to {end_fetch}")
    
    tickers = [row[0] for row in con.execute("SELECT DISTINCT ticker FROM features_cn").fetchall()]
    
    # 演示目的：仅选取前 10 只标的
    tickers = tickers[:10]
    logger.info(f"Syncing top {len(tickers)} tickers for demonstration.")
    
    all_prices = []
    
    for ticker in tickers:
        # 为了计算因子，需要之前的历史数据
        # 我们取前 400 天的数据，以覆盖 252+ 因子窗口
        start_with_buffer = (start_fetch - timedelta(days=400)).strftime('%Y-%m-%d')
        df = fetch_stock_data(ticker, start_with_buffer, end_fetch.strftime('%Y-%m-%d'))
        
        if df is not None and not df.empty:
            df_feat = calculate_features(df)
            # 只保留新增日期的数据
            df_new = df_feat[df_feat['date'] >= start_fetch]
            
            if not df_new.empty:
                # 写入 prices_cn
                con.execute("INSERT INTO prices_cn SELECT ticker, date, open, high, low, close, volume, turn, NULL as ps FROM df_new")
                
                # 写入 features_cn (目前不包含 rank 逻辑，因为 rank 是截面的)
                # rank 可以在入库后通过 SQL 或在主程序中计算
                for _, row in df_new.iterrows():
                    con.execute("""
                        INSERT INTO features_cn (ticker, date, mom_20d, mom_60d, mom_12m_minus_1m, vol_60d_res, sp_ratio, turn_20d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row['ticker'], row['date'], row['mom_20d'], row['mom_60d'], row['mom_12m_minus_1m'], row['vol_60d_res'], row['sp_ratio'], row['turn_20d']))
                
                logger.info(f"Updated {ticker}: {len(df_new)} rows")
        
        time.sleep(0.5) # 频率限制
    
    # 截面 Rank 计算 (更新 features_cn 中的 rank 列)
    dates = con.execute(f"SELECT DISTINCT date FROM features_cn WHERE date >= '{start_fetch}'").fetchall()
    for d_row in dates:
        d = d_row[0]
        logger.info(f"Updating cross-sectional ranks for {d}")
        # 这里使用 DuckDB 的开窗函数直接更新
        for feat in ['mom_20d', 'mom_60d', 'mom_12m_minus_1m', 'vol_60d_res', 'sp_ratio', 'turn_20d']:
            con.execute(f"""
                UPDATE features_cn 
                SET {feat}_rank = sub.r
                FROM (
                    SELECT ticker, date, percent_rank() OVER (PARTITION BY date ORDER BY {feat}) as r
                    FROM features_cn
                    WHERE date = '{d}'
                ) sub
                WHERE features_cn.ticker = sub.ticker AND features_cn.date = sub.date
            """)

    con.close()
    logger.info("Sync completed.")

if __name__ == "__main__":
    sync()
