---
description: 20260406 自动化数据同步与每日预测工作流
---

# 🐌 Snail-Shell 20260406 自动化工作流

该工作流用于 A 股市场的每日增量数据同步、模型预测及报告生成。

## 准备工作
// turbo
1. 环境检查与依赖安装：
   ```bash
   pip install -r requirements.txt
   ```

## 执行流程

2. **数据同步 (Goal 1)**：
   运行同步脚本，从 `akshare` 获取最新行情并更新数据库 `features_cn` 表。
   // turbo
   ```bash
   python scripts/sync_data.py
   ```

3. **每日预测 (Goal 2)**：
   执行预测脚本，计算最新交易日的 Snail-1.0 预测值，并进行螺旋监控分析。
   // turbo
   ```bash
   python scripts/daily_predict.py
   ```

4. **查看报告**：
   预测完成后，报告将保存在 `reports/daily/` 目录下。
   // turbo
   ```bash
   ls -alt reports/daily/
   ```

## 维护说明
- 若 `akshare` 接口失效，请检查网络连接或更新库版本。
- 若数据库出现特征不连贯，请运行 `python main.py check` 进行健康检查。
