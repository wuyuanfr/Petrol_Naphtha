"""
评估与可视化：指标计算、预测对比图、残差分析、SHAP特征重要性。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font="DejaVu Sans")


# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.any() else np.nan
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "MAPE(%)": round(mape, 4)}


# =====================================================================
def plot_predictions(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                     save_dir: str, title_prefix: str = ""):
    fig, axes = plt.subplots(len(predictions), 1,
                             figsize=(14, 4 * len(predictions)), squeeze=False)
    for ax, (name, y_pred) in zip(axes[:, 0], predictions.items()):
        n = min(len(y_true), len(y_pred))
        ax.plot(y_true[:n], label="实测值", color="steelblue", linewidth=1.5)
        ax.plot(y_pred[:n], label="预测值", color="tomato", linewidth=1.5, linestyle="--")
        m = compute_metrics(y_true[:n], y_pred[:n])
        ax.set_title(f"{title_prefix}{name}  R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}°C")
        ax.legend()
        ax.set_ylabel("终馏点 (°C)")
    axes[-1, 0].set_xlabel("样本序号")
    plt.tight_layout()
    path = Path(save_dir) / f"{title_prefix}predictions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"预测对比图保存: {path}")


def plot_scatter(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                 save_dir: str, title_prefix: str = ""):
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        n = min(len(y_true), len(y_pred))
        ax.scatter(y_true[:n], y_pred[:n], alpha=0.5, s=20, color="steelblue")
        lims = [min(y_true[:n].min(), y_pred[:n].min()),
                max(y_true[:n].max(), y_pred[:n].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        m = compute_metrics(y_true[:n], y_pred[:n])
        ax.set_title(f"{name}\nR²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}")
        ax.set_xlabel("实测值 (°C)")
        ax.set_ylabel("预测值 (°C)")
    plt.tight_layout()
    path = Path(save_dir) / f"{title_prefix}scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"散点图保存: {path}")


def plot_residuals(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                   save_dir: str, title_prefix: str = ""):
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        n = min(len(y_true), len(y_pred))
        res = y_true[:n] - y_pred[:n]
        ax.hist(res, bins=30, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(f"{name} 残差分布\n均值={res.mean():.2f}  σ={res.std():.2f}")
        ax.set_xlabel("残差 (°C)")
    plt.tight_layout()
    path = Path(save_dir) / f"{title_prefix}residuals.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"残差图保存: {path}")


def plot_feature_importance(model, feature_names: list[str], save_dir: str,
                            model_name: str = "", top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx[::-1]], imp[idx[::-1]], color="steelblue")
    ax.set_title(f"{model_name} 特征重要度 (Top {top_n})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = Path(save_dir) / f"{model_name}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"特征重要度图保存: {path}")


def plot_shap(model, X_test: np.ndarray, feature_names: list[str],
              save_dir: str, model_name: str = ""):
    try:
        import shap
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_test[:200])
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names,
                          show=False, plot_type="bar")
        path = Path(save_dir) / f"{model_name}_shap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP图保存: {path}")
    except Exception as e:
        logger.warning(f"SHAP绘图失败: {e}")


def save_report(metrics_all: dict, save_dir: str):
    path = Path(save_dir) / "evaluation_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告保存: {path}")

    # 控制台打印
    print("\n" + "=" * 50)
    print("          模型评估结果")
    print("=" * 50)
    for model_name, metrics in metrics_all.items():
        print(f"\n【{model_name}】")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print("=" * 50)
