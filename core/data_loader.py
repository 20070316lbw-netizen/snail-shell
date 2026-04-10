"""
data_loader.py - 数据加载模块

从QQ_Quant_DB数据库加载A股数据用于模型训练和预测
"""

import duckdb
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import date
import os


class DataLoader:
    """
    数据加载器

    从DuckDB数据库加载A股日频数据
    """

    def __init__(self, db_path: str = "QQ_Quant_DB/quant_lab.duckdb"):
        """
        初始化数据加载器

        参数:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.conn = None

    def connect(self) -> None:
        """连接数据库"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        self.conn = duckdb.connect(self.db_path, read_only=True)

    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def get_features(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        index_group: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取特征数据

        参数:
            ticker: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            index_group: 指数组（可选，如'HS300'或'ZZ500'）

        返回:
            特征数据DataFrame
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM features_cn WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if index_group:
            query += " AND index_group = ?"
            params.append(index_group)

        query += " ORDER BY date, ticker"

        df = self.conn.execute(query, params).fetchdf()
        return df

    def get_news_sentiment(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        获取新闻情感数据

        参数:
            ticker: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        返回:
            新闻情感数据DataFrame
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM sentiment_daily WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date, ticker"

        df = self.conn.execute(query, params).fetchdf()
        return df

    def get_labeled_news(
        self, ticker: Optional[str] = None, news_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取已标注的新闻数据

        参数:
            ticker: 股票代码（可选）
            news_type: 新闻类型（可选）

        返回:
            已标注新闻DataFrame
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM news_labeled WHERE 1=1"
        params = []

        if ticker:
            query += " AND affected_ticker = ?"
            params.append(ticker)
        if news_type:
            query += " AND news_type = ?"
            params.append(news_type)

        query += " ORDER BY labeled_at DESC"

        df = self.conn.execute(query, params).fetchdf()
        return df

    def get_alpha_scores(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        获取Alpha因子得分

        参数:
            ticker: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        返回:
            Alpha得分DataFrame
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM alpha_scores WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date, ticker"

        df = self.conn.execute(query, params).fetchdf()
        return df

    def prepare_training_data(
        self, target_date: date, horizon: int = 1, index_group: str = "HS300"
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        准备训练数据

        参数:
            target_date: 目标日期
            horizon: 预测期数（默认1天）
            index_group: 指数组（默认HS300）

        返回:
            (特征DataFrame, 特征矩阵 X, 标签数组 y) —— 共三个返回值
        """
        if not self.conn:
            self.connect()

        # 获取特征数据
        features_df = self.get_features(
            start_date=target_date, end_date=target_date, index_group=index_group
        )

        if features_df.empty:
            raise ValueError(f"没有找到 {target_date} 的数据")

        # 提取特征列（排除非特征列）
        feature_cols = [
            "mom_20d",
            "mom_60d",
            "mom_12m_minus_1m",
            "vol_60d_res",
            "sp_ratio",
            "turn_20d",
            "mom_20d_rank",
            "mom_60d_rank",
            "mom_12m_minus_1m_rank",
            "vol_60d_res_rank",
            "sp_ratio_rank",
            "turn_20d_rank",
        ]

        X = features_df[feature_cols].values

        # 获取标签（未来收益率）
        # 这里需要从prices表计算未来收益率
        # 暂时使用label_next_month作为示例
        y = features_df["label_next_month"].values

        return features_df, X, y

    def get_available_tickers(
        self, index_group: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """
        获取可用的股票代码列表

        参数:
            index_group: 指数组（可选，如果字段为空则忽略）
            limit: 限制返回数量（可选）

        返回:
            股票代码列表
        """
        if not self.conn:
            self.connect()

        query = "SELECT DISTINCT ticker FROM features_cn"
        params = []

        # 检查index_group字段是否有值
        if index_group:
            # 先检查字段是否有值
            check_query = (
                "SELECT COUNT(*) FROM features_cn WHERE index_group IS NOT NULL"
            )
            count = self.conn.execute(check_query).fetchone()[0]
            if count > 0:
                query += " WHERE index_group = ?"
                params.append(index_group)

        if limit:
            query += " LIMIT ?"
            params.append(int(limit))

        result = self.conn.execute(query, params).fetchall()
        return [row[0] for row in result]

    def get_date_range(self) -> Tuple[date, date]:
        """
        获取数据日期范围

        返回:
            (开始日期, 结束日期) 元组
        """
        if not self.conn:
            self.connect()

        result = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM features_cn"
        ).fetchone()

        return result[0], result[1]


if __name__ == "__main__":
    # 测试数据加载器
    print("Testing DataLoader...")

    with DataLoader() as loader:
        # 获取日期范围
        start_date, end_date = loader.get_date_range()
        print(f"数据日期范围: {start_date} ~ {end_date}")

        # 获取可用股票代码
        tickers = loader.get_available_tickers(index_group="HS300")
        print(f"HS300成分股数量: {len(tickers)}")
        print(f"前5只股票: {tickers[:5]}")

        # 获取特征数据样本
        features = loader.get_features(
            ticker=tickers[0], start_date=date(2024, 8, 1), end_date=date(2024, 8, 31)
        )
        print(f"\n{tickers[0]} 8月特征数据:")
        print(f"  行数: {len(features)}")
        print(f"  列名: {list(features.columns)}")

        # 获取新闻情感数据
        sentiment = loader.get_news_sentiment(
            ticker=tickers[0], start_date=date(2024, 8, 1), end_date=date(2024, 8, 31)
        )
        print(f"\n{tickers[0]} 8月新闻情感数据:")
        print(f"  行数: {len(sentiment)}")

        print("\nDataLoader测试完成!")
