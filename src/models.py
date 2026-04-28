"""
模型定义：
- PLSModel       : 偏最小二乘（工业软测量基线）
- XGBoostModel   : 梯度提升树
- LightGBMModel  : 轻量梯度提升
- LSTMModel      : 长短时记忆网络（时序软测量）
- StackingEnsemble: 元学习器融合
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


# =====================================================================
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray): ...
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def get_params(self) -> dict:
        return {}


# =====================================================================
class PLSModel(BaseModel):
    """
    偏最小二乘回归。
    原理：找潜变量 T = XW，最大化 cov(T, y)。
    适合：高维强共线性（蒸馏塔各层温度高度相关）。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["models"]["pls"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_comp_range = self.cfg.get("n_components_range", [5, 8, 10, 12, 15])
        n_comp_range = [n for n in n_comp_range if n < min(X.shape)]

        tscv = TimeSeriesSplit(n_splits=5)
        best_score, best_n = -np.inf, n_comp_range[0]
        for n in n_comp_range:
            scores = []
            for tr, va in tscv.split(X):
                m = PLSRegression(n_components=n)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[va]).ravel()
                r2 = 1 - np.sum((y[va] - pred) ** 2) / np.sum((y[va] - y[va].mean()) ** 2)
                scores.append(r2)
            mean_r2 = np.mean(scores)
            logger.debug(f"PLS n_comp={n} CV R²={mean_r2:.4f}")
            if mean_r2 > best_score:
                best_score, best_n = mean_r2, n

        logger.info(f"PLS最优 n_components={best_n}  CV R²={best_score:.4f}")
        self.model = PLSRegression(n_components=best_n)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).ravel()

    def get_params(self) -> dict:
        return {"n_components": self.model.n_components}


# =====================================================================
class XGBoostModel(BaseModel):
    """
    XGBoost梯度提升树。
    原理：迭代拟合残差，F_m(x) = F_{m-1}(x) + η·h_m(x)。
    每棵树最小化二阶泰勒展开近似的损失函数。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["models"]["xgboost"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        from xgboost import XGBRegressor
        n = len(y)
        val_size = max(int(n * 0.15), 10)
        X_tr, X_va = X[:-val_size], X[-val_size:]
        y_tr, y_va = y[:-val_size], y[-val_size:]

        self.model = XGBRegressor(
            n_estimators=self.cfg.get("n_estimators", 500),
            max_depth=self.cfg.get("max_depth", 5),
            learning_rate=self.cfg.get("learning_rate", 0.03),
            subsample=self.cfg.get("subsample", 0.8),
            colsample_bytree=self.cfg.get("colsample_bytree", 0.8),
            min_child_weight=self.cfg.get("min_child_weight", 3),
            reg_alpha=self.cfg.get("reg_alpha", 0.1),
            reg_lambda=self.cfg.get("reg_lambda", 1.0),
            early_stopping_rounds=self.cfg.get("early_stopping_rounds", 50),
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        logger.info(f"XGBoost best_ntree={self.model.best_iteration}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


# =====================================================================
class LightGBMModel(BaseModel):
    """
    LightGBM：基于直方图的梯度提升，速度更快，内存更省。
    原理与XGBoost相同，区别在于：叶节点优先生长（leaf-wise）。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["models"]["lightgbm"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        n = len(y)
        val_size = max(int(n * 0.15), 10)
        X_tr, X_va = X[:-val_size], X[-val_size:]
        y_tr, y_va = y[:-val_size], y[-val_size:]

        callbacks = [lgb.early_stopping(self.cfg.get("early_stopping_rounds", 50), verbose=False),
                     lgb.log_evaluation(period=-1)]
        self.model = lgb.LGBMRegressor(
            n_estimators=self.cfg.get("n_estimators", 500),
            max_depth=self.cfg.get("max_depth", 5),
            learning_rate=self.cfg.get("learning_rate", 0.03),
            num_leaves=self.cfg.get("num_leaves", 31),
            subsample=self.cfg.get("subsample", 0.8),
            colsample_bytree=self.cfg.get("colsample_bytree", 0.8),
            min_child_samples=self.cfg.get("min_child_samples", 10),
            reg_alpha=self.cfg.get("reg_alpha", 0.1),
            reg_lambda=self.cfg.get("reg_lambda", 1.0),
            random_state=42,
            verbosity=-1,
        )
        self.model.fit(X_tr, y_tr,
                       eval_set=[(X_va, y_va)],
                       callbacks=callbacks)
        logger.info(f"LightGBM best_iter={self.model.best_iteration_}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


# =====================================================================
class LSTMModel(BaseModel):
    """
    LSTM软测量。
    原理：门控机制（输入门、遗忘门、输出门）使网络选择性记忆过程历史，
    捕捉蒸馏塔的传输时延与动态响应。
    输入形状: (batch, seq_len, n_features)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["models"]["lstm"]
        self.seq_len = self.cfg.get("sequence_length", 12)
        self._trained = False

    def _build_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        seqs, targets = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len:i])
            if y is not None:
                targets.append(y[i])
        seqs = np.array(seqs, dtype=np.float32)
        if y is not None:
            return seqs, np.array(targets, dtype=np.float32)
        return seqs

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        seqs, targets = self._build_sequences(X, y)
        n_feat = seqs.shape[2]

        class _LSTM(nn.Module):
            def __init__(self, n_feat, hidden, n_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                                    batch_first=True, dropout=dropout if n_layers > 1 else 0)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        hidden = self.cfg.get("hidden_size", 64)
        n_layers = self.cfg.get("num_layers", 2)
        dropout = self.cfg.get("dropout", 0.2)
        epochs = self.cfg.get("epochs", 100)
        batch_size = self.cfg.get("batch_size", 32)
        lr = self.cfg.get("learning_rate", 0.001)
        patience = self.cfg.get("patience", 15)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _LSTM(n_feat, hidden, n_layers, dropout).to(device)

        # 时序验证集
        val_size = max(int(len(seqs) * 0.15), 5)
        X_tr = torch.tensor(seqs[:-val_size]).to(device)
        y_tr = torch.tensor(targets[:-val_size]).to(device)
        X_va = torch.tensor(seqs[-val_size:]).to(device)
        y_va = torch.tensor(targets[-val_size:]).to(device)

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=False)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        loss_fn = nn.MSELoss()

        best_val, no_imp, best_state = np.inf, 0, None
        for ep in range(epochs):
            self.net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(X_va), y_va).item()
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val, no_imp = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= patience:
                    logger.info(f"LSTM early stop @ epoch {ep}")
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self._device = device
        self._trained = True
        logger.info(f"LSTM训练完成, best_val_loss={best_val:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        seqs = self._build_sequences(X)
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(seqs).to(self._device)
            preds = self.net(t).cpu().numpy()
        # 前seq_len个点无历史，用第一个预测值填充
        pad = np.full(self.seq_len, preds[0])
        return np.concatenate([pad, preds])


# =====================================================================
class StackingEnsemble(BaseModel):
    """
    Stacking集成：用各基模型预测值作为元特征，训练Ridge元学习器。
    原理：不同模型捕捉不同非线性，元学习器自动权衡。
    """

    def __init__(self, base_models: list[BaseModel], cfg: dict):
        self.base_models = base_models
        self.meta = Ridge(alpha=1.0)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for m in self.base_models:
            p = m.predict(X)
            if len(p) != len(X):   # LSTM pad处理
                p = p[-len(X):]
            preds.append(p)
        return np.column_stack(preds)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 用时序CV生成oof预测作为元特征
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        oof = np.zeros((len(y), len(self.base_models)))

        for i, m in enumerate(self.base_models):
            for tr, va in tscv.split(X):
                m_clone = m.__class__(m.cfg if hasattr(m, "cfg") else {})
                m_clone.fit(X[tr], y[tr])
                p = m_clone.predict(X[va])
                if len(p) != len(va):
                    p = p[-len(va):]
                oof[va, i] = p

        self.meta.fit(oof, y)
        logger.info(f"Stacking meta weights: {self.meta.coef_}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_feat = self._get_meta_features(X)
        return self.meta.predict(meta_feat)
