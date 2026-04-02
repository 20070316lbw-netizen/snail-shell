"""
test_data_loading.py - 数据加载测试示例

展示如何使用DataLoader加载A股数据
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import DataLoader
from config import get_database_path
from datetime import date, datetime
import pandas as pd


def main():
    """主函数"""
    print("=" * 60)
    print("snail-shell 数据加载测试")
    print("=" * 60)

    # 使用配置中的数据库路径
    db_path = get_database_path()
    print(f"\n数据库路径: {db_path}")

    # 创建数据加载器
    with DataLoader(db_path) as loader:
        # 1. 获取基本信息
        print("\n1. 数据库基本信息")
        print("-" * 40)

        start_date, end_date = loader.get_date_range()
        print(f"  日期范围: {start_date} ~ {end_date}")

        tickers = loader.get_available_tickers(limit=20)
        print(f"  股票数量: {len(tickers)} (显示前20只)")
        print(f"  股票代码: {tickers[:10]}...")

        # 2. 获取单只股票的特征数据
        print("\n2. 特征数据示例")
        print("-" * 40)

        ticker = tickers[0]
        print(f"  股票: {ticker}")

        features = loader.get_features(
            ticker=ticker, start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)
        )

        print(f"  数据行数: {len(features)}")
        print(f"  特征列: {list(features.columns)}")

        if not features.empty:
            print("\n  最新数据:")
            latest = features.iloc[-1]
            print(f"    日期: {latest['date']}")
            print(f"    20日动量: {latest['mom_20d']:.4f}")
            print(f"    60日动量: {latest['mom_60d']:.4f}")
            print(f"    波动率残差: {latest['vol_60d_res']:.4f}")
            print(f"    S/P比率: {latest['sp_ratio']:.4f}")
            print(f"    换手率: {latest['turn_20d']:.4f}")
            print(f"    下月标签: {latest['label_next_month']:.4f}")

        # 3. 获取多只股票的最新特征
        print("\n3. 多股票最新特征")
        print("-" * 40)

        latest_date = end_date
        all_features = loader.get_features(start_date=latest_date, end_date=latest_date)

        print(f"  最新日期: {latest_date}")
        print(f"  股票数量: {len(all_features)}")

        if not all_features.empty:
            # 按S/P比率排序
            sorted_df = all_features.sort_values("sp_ratio", ascending=False)
            print("\n  S/P比率最高的5只股票:")
            for row in sorted_df.head().itertuples(index=False):
                print(
                    f"    {row.ticker}: S/P={row.sp_ratio:.4f}, "
                    f"20日动量={row.mom_20d:.4f}"
                )

        # 4. 获取新闻情感数据
        print("\n4. 新闻情感数据")
        print("-" * 40)

        sentiment = loader.get_news_sentiment(
            ticker=ticker, start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)
        )

        print(f"  {ticker} 新闻情感数据:")
        print(f"    行数: {len(sentiment)}")

        if not sentiment.empty:
            print(f"    列名: {list(sentiment.columns)}")
            print(f"    平均情绪分数: {sentiment['avg_score'].mean():.4f}")

        # 5. 获取已标注新闻
        print("\n5. 已标注新闻")
        print("-" * 40)

        labeled_news = loader.get_labeled_news(news_type="financial_news")
        print(f"  财经新闻数量: {len(labeled_news)}")

        if not labeled_news.empty:
            print(f"  标签分布:")
            label_counts = labeled_news["label"].value_counts()
            for label, count in label_counts.items():
                print(f"    {label}: {count}")

        # 6. 获取Alpha得分
        print("\n6. Alpha因子得分")
        print("-" * 40)

        alpha_scores = loader.get_alpha_scores(
            ticker=ticker, start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)
        )

        print(f"  {ticker} Alpha得分:")
        print(f"    行数: {len(alpha_scores)}")

        if not alpha_scores.empty:
            print(f"    HS300得分: {alpha_scores['score_hs300'].mean():.4f}")
            print(f"    ZZ500得分: {alpha_scores['score_zz500'].mean():.4f}")

    print("\n" + "=" * 60)
    print("数据加载测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
