# Petrol Naphtha 软测量预测系统

基于机器学习的石油石脑油终馏点软测量预测流水线，支持 PLS、XGBoost、LightGBM、LSTM 及 Stacking 集成模型。

## 项目结构

```
├── main.py              # 主入口，端到端训练与评估
├── predict.py           # 在线预测接口
├── config.yaml          # 全局配置文件
├── requirements.txt     # 依赖列表
├── src/
│   ├── config.py        # 配置加载
│   ├── data_loader.py   # 数据读取与切分
│   ├── preprocessor.py  # 异常值处理、缺失值填充、滚动特征
│   ├── feature_selector.py  # PLS-VIP / 树重要性 / 联合特征选择
│   ├── models.py        # 模型定义（PLS、XGBoost、LightGBM、LSTM、Stacking）
│   ├── trainer.py       # 训练调度
│   └── evaluator.py     # 指标计算与可视化
├── notebooks/           # 探索性分析与实验 Notebook
└── outputs/
    ├── models/          # 保存的模型文件
    ├── figures/         # 预测图、散点图、SHAP 图
    └── reports/         # 评估报告
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练

```bash
# 使用默认 config.yaml
python main.py

# 指定配置文件
python main.py --config my.yaml

# 跳过 LSTM（速度较慢）
python main.py --no-lstm
```

### 预测

```bash
python predict.py
```

## 配置说明

编辑 `config.yaml` 调整各阶段参数：

| 模块 | 关键参数 |
|------|---------|
| 数据 | `file_pattern`、`target_col`、`bad_value_threshold` |
| 预处理 | `outlier_method`（iqr/zscore/isolation_forest）、`impute_method` |
| 特征选择 | `method`（combined/pls_vip/tree_importance）、`top_k_features` |
| 模型 | 各模型的超参数及 `enabled` 开关 |
| 集成 | `ensemble.method`（stacking/weighted_avg） |

## 模型说明

- **PLS**：偏最小二乘，适合高维共线性数据
- **XGBoost / LightGBM**：梯度提升树，支持 SHAP 可解释性分析
- **LSTM**：时序窗口建模，捕捉动态过程特征
- **Stacking**：以 Ridge 回归作为元学习器融合上述模型

## 输出

训练完成后结果保存在 `outputs/`：
- `models/`：预处理器、特征选择器及各模型文件
- `figures/`：预测曲线、散点图、残差图、特征重要性、SHAP 图
- `reports/`：RMSE、MAE、R²、MAPE 指标报告
