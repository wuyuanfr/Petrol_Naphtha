"""
特征选择：
- PLS VIP（Variable Importance in Projection）：工业软测量标准方法
- 树模型特征重要度（XGBoost）
- 组合策略：PLS VIP + 树模型排名取并集
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cross_decomposition import PLSRegression

logger = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self, cfg: dict):
        self.cfg = cfg["feature_selection"]
        self.selected_indices: list[int] = []
        self.feature_scores: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        method = self.cfg.get("method", "combined")
        top_k = self.cfg.get("top_k_features", 30)

        if method == "pls_vip":
            scores = self._pls_vip(X, y)
        elif method == "tree_importance":
            scores = self._tree_importance(X, y)
        else:  # combined
            vip = self._pls_vip(X, y)
            imp = self._tree_importance(X, y)
            # 归一化后平均
            vip_norm = (vip - vip.min()) / (vip.ptp() + 1e-9)
            imp_norm = (imp - imp.min()) / (imp.ptp() + 1e-9)
            scores = 0.5 * vip_norm + 0.5 * imp_norm

        self.feature_scores = scores
        top_k = min(top_k, X.shape[1])
        self.selected_indices = np.argsort(scores)[::-1][:top_k].tolist()

        if feature_names:
            selected_names = [feature_names[i] for i in self.selected_indices]
            logger.info(f"选择了 {top_k} 个特征，Top-10: {selected_names[:10]}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    # ------------------------------------------------------------------
    def _pls_vip(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_comp = min(self.cfg.get("pls_n_components", 15), X.shape[1], X.shape[0] - 1)
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X, y)

        # VIP = sqrt(p * sum_h(W_h^2 * SS_h) / SS_total)
        W = pls.x_weights_          # (p, n_comp)
        T = pls.x_scores_           # (n, n_comp)
        Q = pls.y_loadings_         # (q, n_comp)

        ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)  # (n_comp,)
        ss_total = np.sum(ss)

        p = X.shape[1]
        W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)
        vip = np.sqrt(p * np.sum(W_norm ** 2 * ss, axis=1) / (ss_total + 1e-9))
        logger.info(f"PLS VIP: {(vip >= 1.0).sum()} 个变量VIP≥1.0")
        return vip

    def _tree_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        scores = model.feature_importances_
        logger.info(f"XGBoost特征重要度: top5={np.sort(scores)[::-1][:5]}")
        return scores
