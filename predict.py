"""
在线推理脚本：加载保存的模型，对新数据做预测。

用法:
    python predict.py --input new_data.csv --model xgboost
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="待预测CSV文件路径")
    p.add_argument("--model", default="ensemble", help="模型名: pls/xgboost/lightgbm/ensemble")
    p.add_argument("--model-dir", default="outputs/models")
    p.add_argument("--output", default="outputs/predictions.csv")
    return p.parse_args()


def predict(args):
    model_dir = Path(args.model_dir)

    prep = joblib.load(model_dir / "preprocessor.pkl")
    sel = joblib.load(model_dir / "feature_selector.pkl")
    model = joblib.load(model_dir / f"{args.model}.pkl")
    selected_names = joblib.load(model_dir / "selected_features.pkl")

    df = pd.read_csv(args.input, sep="\t", encoding="utf-8-sig")
    # 去除目标列（如果存在）
    target_col = "FORMATTED_ENTRY"
    y_true = df.pop(target_col) if target_col in df.columns else None
    product_col = "PRODUCT"
    if product_col in df.columns:
        df = df.drop(columns=[product_col])
    X_raw = df.select_dtypes(include=[np.number])

    # 预处理
    X_prep = prep.transform(X_raw)
    X_sel = sel.transform(X_prep)

    preds = model.predict(X_sel)
    result = pd.DataFrame({"prediction": preds})
    if y_true is not None:
        result["actual"] = y_true.values[:len(preds)]
        result["error"] = result["actual"] - result["prediction"]

    result.to_csv(args.output, index=False)
    logger.info(f"预测结果保存: {args.output}")
    if y_true is not None:
        n = min(len(y_true), len(preds))
        from src.evaluator import compute_metrics
        m = compute_metrics(y_true.values[:n], preds[:n])
        print(f"评估指标: {m}")


if __name__ == "__main__":
    predict(parse_args())
