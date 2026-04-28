"""
数据加载与基础质量检查。
- 识别DCS坏值（-99.x / -100 等）并置NaN
- 自动检测分隔符（制表符或逗号）
- 对PRODUCT列做独热编码备用
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """
    返回: X (特征), y (目标), product_series (产品标签，可为None)
    """
    pattern = cfg["data"]["file_pattern"]
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")

    frames = []
    for f in files:
        sep = cfg["data"].get("separator", "\t")
        df = pd.read_csv(f, sep=sep, encoding="utf-8-sig")
        frames.append(df)
        logger.info(f"加载文件: {f}  shape={df.shape}")

    data = pd.concat(frames, ignore_index=True)
    logger.info(f"合并后 shape={data.shape}")

    target_col = cfg["data"]["target_col"]
    product_col = cfg["data"].get("product_col")
    bad_thresh = cfg["data"].get("bad_value_threshold", -50.0)

    if target_col not in data.columns:
        raise ValueError(f"目标列 '{target_col}' 不在数据中，现有列: {list(data.columns)}")

    # 替换DCS坏值
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    bad_mask = data[numeric_cols] < bad_thresh
    n_bad = bad_mask.values.sum()
    data[numeric_cols] = data[numeric_cols].where(~bad_mask, other=np.nan)
    logger.info(f"检测到 {n_bad} 个DCS坏值（< {bad_thresh}），已置NaN")

    # 分离目标
    y = data[target_col].copy()
    drop_cols = [target_col]

    # 处理产品列
    product_series = None
    if product_col and product_col in data.columns:
        product_series = data[product_col].copy()
        drop_cols.append(product_col)

    X = data.drop(columns=drop_cols)
    # 仅保留数值列
    X = X.select_dtypes(include=[np.number])

    # 删除目标为NaN的行
    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    if product_series is not None:
        product_series = product_series[valid].reset_index(drop=True)

    logger.info(f"有效样本: {len(y)}  特征数: {X.shape[1]}")
    logger.info(f"目标 {target_col}: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    return X, y, product_series


def split_by_time(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """按时间顺序划分训练/测试集（不随机打乱）。"""
    n = len(y)
    split = int(n * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
