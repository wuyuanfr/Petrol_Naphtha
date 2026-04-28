"""
训练流程编排：时序交叉验证 + 模型选择 + 最终训练。
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .models import LightGBMModel, LSTMModel, PLSModel, XGBoostModel, StackingEnsemble
from .evaluator import compute_metrics

logger = logging.getLogger(__name__)


def cross_validate(model_cls, model_cfg: dict, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = []
    for fold, (tr, va) in enumerate(tscv.split(X)):
        m = model_cls(model_cfg)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        if len(pred) != len(va):
            pred = pred[-len(va):]
        metrics = compute_metrics(y[va], pred)
        all_metrics.append(metrics)
        logger.debug(f"Fold {fold+1}: {metrics}")

    avg = {k: round(float(np.mean([m[k] for m in all_metrics])), 4) for k in all_metrics[0]}
    std = {k: round(float(np.std([m[k] for m in all_metrics])), 4) for k in all_metrics[0]}
    return {"mean": avg, "std": std}


def train_all(cfg: dict, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str] | None = None) -> dict:
    """训练所有启用的模型，返回 {name: model} 字典。"""
    models = {}
    n_splits = cfg["training"].get("n_splits", 5)

    model_map = {
        "pls": PLSModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "lstm": LSTMModel,
    }

    for key, cls in model_map.items():
        if not cfg["models"].get(key, {}).get("enabled", True):
            logger.info(f"跳过 {key.upper()} (disabled)")
            continue
        logger.info(f"--- 训练 {key.upper()} ---")
        cv_result = cross_validate(cls, cfg, X_train, y_train, n_splits)
        logger.info(f"{key.upper()} CV: {cv_result['mean']}")

        # 用全量训练集最终训练
        m = cls(cfg)
        m.fit(X_train, y_train)
        models[key] = m

    # Stacking集成
    if cfg["models"].get("ensemble", {}).get("enabled", True) and len(models) >= 2:
        logger.info("--- 训练 STACKING ENSEMBLE ---")
        base = list(models.values())
        ensemble = StackingEnsemble(base, cfg)
        # 重新在训练集上拟合元学习器
        meta_feats = np.column_stack([
            _align(m.predict(X_train), len(X_train)) for m in base
        ])
        from sklearn.linear_model import Ridge
        ensemble.meta.fit(meta_feats, y_train)
        models["ensemble"] = ensemble

    return models


def _align(pred: np.ndarray, n: int) -> np.ndarray:
    if len(pred) == n:
        return pred
    if len(pred) > n:
        return pred[-n:]
    return np.concatenate([np.full(n - len(pred), pred[0]), pred])
