import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

import joblib

# ========= 1) 读数据 =========
DATA_PATH = r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\3_samples\point1__samples_ml_ready.csv"   # 改成你的路径
df = pd.read_csv(DATA_PATH)

# label 列名（如果你没改过就是 label）
LABEL_COL = "label"

# X / y
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL].astype(int)

print("样本数:", len(df))
print("特征数:", X.shape[1])
print("正样本(1)数量:", int((y == 1).sum()), "负样本(0)数量:", int((y == 0).sum()))

# ========= 2) 划分训练/测试（分层，避免比例失衡） =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# ========= 3) 训练 Random Forest =========
rf = RandomForestClassifier(
    n_estimators=500,        # 树数量
    max_depth=None,          # 不限制深度（可后续调参）
    min_samples_leaf=5,      # 叶子最少样本，防过拟合
    max_features="sqrt",     # 常用设置
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"  # 类别不平衡时更稳（即使你是1:1也不坏）
)

rf.fit(X_train, y_train)

# ========= 4) 评估 =========
# 概率（用于 AUC）
proba_test = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
print("\nROC-AUC:", round(auc, 4))

# 默认阈值 0.5 的分类结果
y_pred = (proba_test >= 0.5).astype(int)

print("\nConfusion Matrix (阈值=0.5):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ========= 5) 特征重要性（论文常用） =========
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 20 Feature Importances:")
print(importances.head(20))

# 保存重要性到文件（可选）
importances.to_csv(r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\4_weights\rf_feature_importance.csv", encoding="utf-8-sig")
joblib.dump(rf, r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\4_weights\rf_model.joblib")
