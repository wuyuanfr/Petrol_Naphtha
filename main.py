"""
主入口：端到端运行软测量预测流水线。

用法:
    python main.py                      # 使用 config.yaml 默认配置
    python main.py --config my.yaml     # 指定配置文件
    python main.py --no-lstm            # 跳过LSTM（速度较慢）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import joblib

from src.config import load_config, ensure_dirs
from src.data_loader import load_dataset, split_by_time
from src.preprocessor import Preprocessor
from src.feature_selector import FeatureSelector
from src.trainer import train_all
from src.evaluator import (
    compute_metrics,
    plot_predictions,
    plot_scatter,
    plot_residuals,
    plot_feature_importance,
    plot_shap,
    save_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--no-lstm", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_lstm:
        cfg["models"]["lstm"]["enabled"] = False
    ensure_dirs(cfg)

    fig_dir = cfg["paths"]["figure_dir"]
    model_dir = cfg["paths"]["model_dir"]
    report_dir = cfg["paths"]["report_dir"]

    # ── 1. 加载数据 ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: 加载数据")
    X_raw, y, product = load_dataset(cfg)
    logger.info(f"原始特征: {X_raw.shape}  目标范围: [{y.min():.1f}, {y.max():.1f}] °C")

    # ── 2. 训练/测试时序切分 ─────────────────────────────────────────
    test_size = cfg["training"].get("test_size", 0.2)
    X_tr_raw, X_te_raw, y_train, y_test = split_by_time(X_raw, y, test_size)
    logger.info(f"训练集: {len(y_train)}  测试集: {len(y_test)}")

    # ── 3. 预处理 ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: 预处理")
    prep = Preprocessor(cfg)
    X_train_prep = prep.fit_transform(X_tr_raw, y_train)
    # 为保持时序，transform测试集时不泄露信息
    X_test_prep_df = X_te_raw.copy()
    for c in prep.kept_cols_before_rolling:
        if c not in X_test_prep_df.columns:
            X_test_prep_df[c] = np.nan
    X_test_prep_df = X_test_prep_df[prep.kept_cols_before_rolling]
    from src.preprocessor import _add_rolling
    if cfg["preprocessing"].get("add_rolling_features", False):
        X_test_prep_df = _add_rolling(X_test_prep_df, cfg["preprocessing"].get("rolling_windows", [3, 6]))
        X_test_prep_df = X_test_prep_df[prep.kept_cols]
    else:
        X_test_prep_df = X_test_prep_df[[c for c in prep.kept_cols if c in X_test_prep_df.columns]]
    X_test_prep = prep.scaler.transform(X_test_prep_df.fillna(0))

    # ── 4. 特征选择 ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: 特征选择")
    sel = FeatureSelector(cfg)
    X_train = sel.fit_transform(X_train_prep, y_train.values,
                                feature_names=prep.kept_cols)
    X_test = sel.transform(X_test_prep)
    selected_names = [prep.kept_cols[i] for i in sel.selected_indices
                      if i < len(prep.kept_cols)]
    logger.info(f"选择后维度: train={X_train.shape}  test={X_test.shape}")

    # ── 5. 训练 ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4: 模型训练")
    models = train_all(cfg, X_train, y_train.values, feature_names=selected_names)

    # ── 6. 评估 ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 5: 评估")
    y_true = y_test.values
    predictions: dict[str, np.ndarray] = {}
    metrics_all: dict[str, dict] = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        n = min(len(y_true), len(pred))
        pred = pred[-n:]
        predictions[name] = pred
        metrics_all[name] = compute_metrics(y_true[:n], pred)

    # ── 7. 可视化 ────────────────────────────────────────────────────
    logger.info("Step 6: 可视化")
    plot_predictions(y_true, predictions, fig_dir, title_prefix="test_")
    plot_scatter(y_true, predictions, fig_dir, title_prefix="test_")
    plot_residuals(y_true, predictions, fig_dir, title_prefix="test_")

    for name in ("xgboost", "lightgbm"):
        if name in models:
            plot_feature_importance(models[name], selected_names, fig_dir, model_name=name)
            plot_shap(models[name], X_test, selected_names, fig_dir, model_name=name)

    save_report(metrics_all, report_dir)

    # ── 8. 保存模型 ──────────────────────────────────────────────────
    logger.info("Step 7: 保存模型")
    joblib.dump(prep, Path(model_dir) / "preprocessor.pkl")
    joblib.dump(sel, Path(model_dir) / "feature_selector.pkl")
    for name, model in models.items():
        if name != "lstm":
            joblib.dump(model, Path(model_dir) / f"{name}.pkl")
        else:
            import torch
            torch.save(model.net.state_dict(), Path(model_dir) / "lstm_state.pt")
    joblib.dump(selected_names, Path(model_dir) / "selected_features.pkl")

    logger.info("流水线完成！结果保存在 outputs/")


if __name__ == "__main__":
    main()
