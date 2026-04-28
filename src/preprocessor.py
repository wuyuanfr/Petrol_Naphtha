"""
预处理流水线：
1. 低方差过滤
2. 异常值检测与处理（IQR / Z-score / Isolation Forest）
3. 缺失值插补
4. 高相关冗余列删除
5. 滑动统计特征（捕捉过程动态）
6. 标准化
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg["preprocessing"]
        self.scaler = StandardScaler()
        self.kept_cols: list[str] = []
        self.drop_cols_variance: list[str] = []
        self.drop_cols_corr: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        X = X.copy()
        X = self._remove_low_variance(X, fit=True)
        X = self._handle_outliers(X, fit=True)
        X = self._impute(X, fit=True)
        X = self._remove_high_corr(X, fit=True)
        if self.cfg.get("add_rolling_features", False):  # type: ignore[attr-defined]
            X = _add_rolling(X, self.cfg.get("rolling_windows", [3, 6]))
        X_arr = self.scaler.fit_transform(X)
        self.kept_cols = list(X.columns)
        self._fitted = True
        logger.info(f"预处理后特征数: {len(self.kept_cols)}")
        return X_arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor尚未fit")
        X = X.copy()
        # 仅保留训练时确定的列（补全缺失列为0）
        for c in self.kept_cols_before_rolling:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.kept_cols_before_rolling]
        X = self._handle_outliers(X, fit=False)
        X = self._impute(X, fit=False)
        if self.cfg.get("add_rolling_features", False):
            X = _add_rolling(X, self.cfg.get("rolling_windows", [3, 6]))
        X = X[self.kept_cols]  # 对齐列顺序
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    def _remove_low_variance(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        thresh = self.cfg.get("variance_threshold", 0.001)
        if fit:
            var = X.var(numeric_only=True)
            self.drop_cols_variance = list(var[var < thresh].index)
        drop = [c for c in self.drop_cols_variance if c in X.columns]
        logger.info(f"低方差过滤: 删除 {len(drop)} 列")
        return X.drop(columns=drop)

    def _handle_outliers(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        method = self.cfg.get("outlier_method", "iqr")
        if method == "iqr":
            factor = self.cfg.get("outlier_iqr_factor", 3.0)
            if fit:
                self._q1 = X.quantile(0.25)
                self._q3 = X.quantile(0.75)
                self._iqr = self._q3 - self._q1
            lo = self._q1 - factor * self._iqr
            hi = self._q3 + factor * self._iqr
            X = X.clip(lower=lo, upper=hi, axis=1)
        elif method == "zscore":
            thresh = self.cfg.get("outlier_zscore_threshold", 4.0)
            if fit:
                self._mean = X.mean()
                self._std = X.std().replace(0, 1)
            z = (X - self._mean) / self._std
            X = X.where(z.abs() < thresh, other=np.nan)
        elif method == "isolation_forest":
            if fit:
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(contamination=0.05, random_state=42)
                mask = clf.fit_predict(X.fillna(X.median())) == -1
                self._if_mask = mask
                logger.info(f"IsolationForest检测异常: {mask.sum()} 行")
            # 异常行置NaN（由impute后续处理）
            if fit and self._if_mask.any():
                X[self._if_mask] = np.nan
        return X

    def _impute(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        method = self.cfg.get("impute_method", "median")
        if method == "median":
            if fit:
                self._fill_values = X.median()
            X = X.fillna(self._fill_values)
        elif method == "mean":
            if fit:
                self._fill_values = X.mean()
            X = X.fillna(self._fill_values)
        elif method == "knn":
            from sklearn.impute import KNNImputer
            if fit:
                self._knn_imputer = KNNImputer(n_neighbors=5)
                arr = self._knn_imputer.fit_transform(X)
            else:
                arr = self._knn_imputer.transform(X)
            X = pd.DataFrame(arr, columns=X.columns, index=X.index)
        return X

    def _remove_high_corr(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        thresh = self.cfg.get("correlation_threshold", 0.97)
        if fit:
            corr = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            self.drop_cols_corr = [c for c in upper.columns if (upper[c] > thresh).any()]
            # 保存rolling之前的列列表
            self.kept_cols_before_rolling = [c for c in X.columns if c not in self.drop_cols_corr]
        drop = [c for c in self.drop_cols_corr if c in X.columns]
        logger.info(f"高相关过滤: 删除 {len(drop)} 列")
        return X.drop(columns=drop)


# ------------------------------------------------------------------
def _add_rolling(X: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    frames = [X]
    for w in windows:
        rolled_mean = X.rolling(w, min_periods=1).mean()
        rolled_mean.columns = [f"{c}_rmean{w}" for c in X.columns]
        rolled_std = X.rolling(w, min_periods=1).std().fillna(0)
        rolled_std.columns = [f"{c}_rstd{w}" for c in X.columns]
        frames.extend([rolled_mean, rolled_std])
    return pd.concat(frames, axis=1)
