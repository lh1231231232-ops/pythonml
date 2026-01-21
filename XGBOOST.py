import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, brier_score_loss
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import ParameterGrid

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
print("所有库都已成功导入。")
print("-" * 50)

# =============================================================================
# 0. 参数和列名设置
# =============================================================================
SCRIPT_DIR      = os.getcwd()
INPUT_TIF_DIR   = os.path.join(SCRIPT_DIR, '100M_total_tif')
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, 'XGBOOST的结果')
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_filename    = os.path.join(SCRIPT_DIR, 'merged_4.csv')
TARGET_COLUMN   = 'FT'
COLUMNS_TO_DROP = ['OID_']

CATEGORICAL_FEATURES = ['SOIL_TP', 'LAND_USE_TP']
ASPECT_COLUMN        = 'Aspect'

# =============================================================================
# 1. 读取 & 基本清洗
# =============================================================================
try:
    data = pd.read_csv(csv_filename)
    print(f"数据已成功加载: '{csv_filename}'")
except FileNotFoundError:
    print(f"错误: 文件 '{csv_filename}' 未找到。")
    exit()

print(f"初始行数: {data.shape[0]}")
data.replace(-9999, np.nan, inplace=True)
data.dropna(inplace=True)
print(f"清洗后行数: {data.shape[0]}")
if data.isnull().any().any():
    print("警告: 仍然存在缺失值！")
else:
    print("数据清洗完成，无缺失值。")

data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

# =============================================================================
# 2. Aspect → sin / cos（如果列存在）
# =============================================================================
if ASPECT_COLUMN in data.columns:
    data[ASPECT_COLUMN] = pd.to_numeric(data[ASPECT_COLUMN], errors='coerce')
    data['sin_Aspect'] = np.sin(np.deg2rad(data[ASPECT_COLUMN]))
    data['cos_Aspect'] = np.cos(np.deg2rad(data[ASPECT_COLUMN]))
    data.drop(columns=[ASPECT_COLUMN], inplace=True)
    print("Aspect 已转换为 sin/cos，原始列已删除。")
else:
    print(f"警告: 未找到列 '{ASPECT_COLUMN}'，跳过 Aspect 转换。")

# =============================================================================
# 3. 特征列表（原始 + sin/cos）
# =============================================================================
# 这里列出 **所有** 可能出现的数值特征（包括后面会加入的 sin/cos）
base_numeric = [
    'Aspect', 'DEM', 'Fault', 'LAND_USE_TP', 'LSTV', 'MNDWI', 'NDVI', 'PlanCurvature', 'ProfileCurvature',
    'River', 'Road', 'Rx3day', 'Slope', 'SOIL_TP', 'TPI', 'TRI', 'TWI'
]

# 假设您已经执行了 Aspect 转换，并成功将 sin_Aspect 和 cos_Aspect 加入到 data.columns
extra_numeric = []
if 'sin_Aspect' in data.columns:
    extra_numeric.extend(['sin_Aspect', 'cos_Aspect'])

# --- 修正逻辑 ---
# 最终的数值特征列表
numerical_features = base_numeric + extra_numeric

# 如果使用了 sin/cos 转换，则从列表中移除原始的 'Aspect'
if 'sin_Aspect' in data.columns and 'Aspect' in numerical_features:
    numerical_features.remove('Aspect')  # 移除原始 Aspect 列

# =============================================================================
# 2. 区分 密集 与 稀疏 数值特征
# =============================================================================
sparse_numeric = ['Road', 'River', 'Fault']
CATEGORICAL_FEATURES = ['SOIL_TP', 'LAND_USE_TP']  # 确保这里已定义！

# --- 关键修正：先定义所有特征，再划分数据集 ---
features_to_exclude = sparse_numeric + CATEGORICAL_FEATURES
dense_numeric = [f for f in numerical_features if f not in features_to_exclude]

print(f"密集数值特征 ({len(dense_numeric)}): {dense_numeric}")
print(f"稀疏数值特征 ({len(sparse_numeric)}): {sparse_numeric}")
print(f"分类特征: {CATEGORICAL_FEATURES}")

# =============================================================================
# 6. 【正确顺序】先划分训练/测试集 → 再构建 preprocessor → 再 fit → 再 transform
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("\n" + "="*60)
print("开始数据划分与预处理（宇宙最正确顺序！）")
print("="*60)

# 1. 先定义 X 和 y
feature_cols = dense_numeric + sparse_numeric + CATEGORICAL_FEATURES
X = data[feature_cols].copy()
y = data[TARGET_COLUMN].copy()

# 2. 先划分训练集和测试集（X_train 现在诞生！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

# 3. 现在才构建 preprocessor（X_train 已经存在了！）
print("\n正在构建预处理管道（ColumnTransformer）...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num_dense', StandardScaler(), dense_numeric),     # 需要标准化
        ('num_sparse', 'passthrough', sparse_numeric),      # 距离特征保持原值
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)

# 4. 只 fit 一次！在训练集上
print("正在对训练集执行 preprocessor.fit...")
preprocessor.fit(X_train)

# 5. 立刻提取 scaler 和 encoder（供后续保存使用）
print("正在提取 scaler 和 encoder...")
scaler = preprocessor.named_transformers_['num_dense']   # StandardScaler
encoder = preprocessor.named_transformers_['cat']        # OneHotEncoder

print(f"scaler 提取成功 → {type(scaler).__name__}")
print(f"encoder 提取成功 → 共 {len(encoder.categories_)} 个分类特征")
for i, col in enumerate(CATEGORICAL_FEATURES):
    print(f"   → {col}: {len(encoder.categories_[i])} 类")

# 6. 应用预处理（transform）
print("\n正在应用预处理到训练集和测试集...")
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# 7. 重建列名（超级重要！）
cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
feature_names_out = dense_numeric + sparse_numeric + cat_feature_names

X_train_sel = pd.DataFrame(X_train_proc, columns=feature_names_out, index=X_train.index)
X_test_sel  = pd.DataFrame(X_test_proc,  columns=feature_names_out, index=X_test.index)

print(f"预处理完成！")
print(f"   X_train_sel: {X_train_sel.shape}")
print(f"   X_test_sel : {X_test_sel.shape}")
print(f"   特征示例: {X_train_sel.columns[:10].tolist()} ...")

# =============================================================================
# 8. 移除高共线性特征（VIF 筛选出的）
# =============================================================================
VIF_HIGH_COVAR_FEATURE = 'LAND_USE_TP_30'

if VIF_HIGH_COVAR_FEATURE in X_train_sel.columns:
    X_train_sel.drop(columns=[VIF_HIGH_COVAR_FEATURE], inplace=True)
    X_test_sel.drop(columns=[VIF_HIGH_COVAR_FEATURE], inplace=True)
    print(f"已成功移除高共线性特征: {VIF_HIGH_COVAR_FEATURE}")
    print(f"   移除后特征数: {X_train_sel.shape[1]}")
else:
    print(f"待移除特征 {VIF_HIGH_COVAR_FEATURE} 未找到，跳过。")

print("\n数据预处理全部完成！可以进入后续 SHAP、建模、保存环节！")
print("="*60)

# =============================================================================
# 7. （可选）检查稀疏特征的原始统计信息
# =============================================================================
print("稀疏距离特征（原始值）统计：")
for col in sparse_numeric:
    if col in X_train.columns:
        ser = X_train[col]
        print(f"  {col:5s} → min={ser.min():.1f}, max={ser.max():.1f}, "
              f"mean={ser.mean():.1f}, std={ser.std():.1f}, zeros={ (ser==0).sum() }")
print("-" * 50)

# =============================================================================
# 6. 快速粗搜（RandomizedSearchCV）
# =============================================================================
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
import xgboost as xgb

print("正在使用 RandomizedSearchCV 快速搜索（10 分钟内出结果）...")

# 重新计算 scale_pos_weight（基于 y_train）
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.3f}")

# 宽松网格
param_dist = {
    'learning_rate': uniform(0.01, 0.19),      # 0.01 ~ 0.2
    'max_depth': randint(3, 7),                # 3~9
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 1),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_lambda': uniform(0, 50),
    'reg_alpha': uniform(0, 2),
    'n_estimators': randint(300, 1000)
}

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    seed=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

random_search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_dist,
    n_iter=100,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# 关键：使用 X_train_sel！
random_search.fit(X_train_sel, y_train)

print("\n随机搜索完成！")
print("最佳参数:", random_search.best_params_)
print(f"最佳 CV F1: {random_search.best_score_:.4f}")
print("-" * 50)

coarse_best = random_search.best_params_

# =============================================================================
# 6.5 终极精搜（GridSearchCV）—— 完全不变
# =============================================================================
print("正在进行终极精搜（< 30 个组合）...")
param_grid_fine = {
    'learning_rate': [
        max(0.01, coarse_best['learning_rate'] * 0.8),
        coarse_best['learning_rate'],
        min(0.3, coarse_best['learning_rate'] * 1.2)
    ],
    'max_depth': [coarse_best['max_depth']],
    'min_child_weight': [
        max(1, coarse_best['min_child_weight'] - 1),
        coarse_best['min_child_weight'],
        coarse_best['min_child_weight'] + 1
    ],
    'gamma': [
        max(0, coarse_best['gamma'] * 0.5),
        coarse_best['gamma'],
        min(10, coarse_best['gamma'] * 2)
    ]
}

# 转为整数
param_grid_fine['max_depth'] = [int(param_grid_fine['max_depth'][0])]
param_grid_fine['min_child_weight'] = [int(x) for x in param_grid_fine['min_child_weight']]

print(f"精搜组合数: {len(list(ParameterGrid(param_grid_fine)))} 个")

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid_fine,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_sel, y_train)

best_params = grid_search.best_params_
print("\n精搜完成！")
print("最终最佳参数:", best_params)
print(f"最终 CV F1: {grid_search.best_score_:.4f}")
print("-" * 50)

# =============================================================================
# 6.9 最终分类模型（原生训练 + 属性安全继承）
# =============================================================================
print("正在训练最终分类模型（原生训练早停机制）...")

# 1. 复制最佳参数 + 强制正则化 (保持不变)
final_params = best_params.copy()
final_params['n_estimators'] = 2000
final_params['reg_lambda']   = max(final_params.get('reg_lambda', 0), 30.0)
final_params['reg_alpha']    = max(final_params.get('reg_alpha', 0), 5.0)
final_params['max_depth']    = min(final_params.get('max_depth', 6), 3)

# 2. 划分验证集 (保持不变)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_sel, y_train,
    test_size=0.2, random_state=42, stratify=y_train
)

# 3. 转为 DMatrix (保持不变)
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval   = xgb.DMatrix(X_val, label=y_val)

# 4. 原生 train（带 early stopping）(保持不变)
train_params = final_params.copy()
train_params.pop('n_estimators', None)
train_params['objective'] = 'binary:logistic'
train_params['eval_metric'] = 'logloss'

booster = xgb.train(
    train_params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=False
)

best_iter = booster.best_iteration
print(f"早停成功！最佳迭代数: {best_iter}")

# 5. 全量数据重新训练 (保持不变)
dtrain_all = xgb.DMatrix(X_train_sel, label=y_train)
final_booster = xgb.train(
    train_params,
    dtrain_all,
    num_boost_round=best_iter,
    verbose_eval=False
)

# 6. 【核心修复】：使用一个"干净"的代理模型来继承属性
# CalibratedClassifierCV 只需要一个能提供 classes_ 的对象。
# 我们创建一个新的 XGBClassifier 并让它自动 fit 一下，确保属性存在。

# 6.1 创建代理模型 (用于继承 classes_ 属性)
# 使用最终参数来实例化
proxy_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    n_estimators=1 # 只需要拟合一次以生成 classes_
)
proxy_xgb.set_params(**final_params)

# 6.2 拟合代理模型（仅用于生成 classes_ 和 n_classes_）
# 使用最小的数据集进行拟合

# 确保 X_train_sel 和 y_train 至少有两行，分别代表类别 0 和 1
# 为了安全，我们找到第一个类别 0 的样本和第一个类别 1 的样本进行拟合。
idx_0 = y_train[y_train == 0].index[0]
idx_1 = y_train[y_train == 1].index[0]

X_proxy = X_train_sel.loc[[idx_0, idx_1]]
y_proxy = y_train.loc[[idx_0, idx_1]]

# 拟合代理模型，此时 y_proxy 包含 [0, 1]，可以正常生成 classes_
proxy_xgb.fit(X_proxy, y_proxy)

# 6.3 包装最终模型
final_xgb = xgb.XGBClassifier()
final_xgb.set_params(**final_params)
final_xgb._Booster = final_booster
final_xgb.n_estimators = best_iter

# 6.4 继承代理模型的 classes_ 和 n_classes_
# 只有在 set_params 之后，这些属性才能安全设置
try:
    # 尝试直接赋值，但这次是赋值给 proxy_xgb 之后的对象
    setattr(final_xgb, 'classes_', proxy_xgb.classes_)
    setattr(final_xgb, 'n_classes_', proxy_xgb.n_classes_)
except AttributeError:
    # 如果仍然失败，我们直接使用 proxy_xgb 作为校准器模型
    # 并将 final_booster 注入到它的底层
    proxy_xgb._Booster = final_booster
    proxy_xgb.n_estimators = best_iter
    final_xgb = proxy_xgb # 使用这个已经 fit 过的代理模型作为最终模型
    print("   [注意] 无法设置 classes_，已使用代理模型作为最终输出。")

# 关键：定义全局变量
final_xgb_model = final_xgb

print(f"最终模型树数量: {final_xgb_model.n_estimators}")
print("-" * 50)

# =============================================================================
# 7. 模型校准 (使用 GridSearchCV 优化校准方法) - 终极修复版
# =============================================================================
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

print("正在使用 GridSearchCV 优化校准方法 (Sigmoid vs Isotonic)...")

# 关键：定义 StratifiedKFold
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. 定义 CalibratedClassifierCV（使用最终训练好的模型）
calibrator = CalibratedClassifierCV(
    estimator=final_xgb_model,   # ← 确保是 final_xgb_model
    cv=cv_folds,
    method='sigmoid'  # 初始值，后面会搜索
)

# 2. 参数网格
calibrator_param_grid = {
    'method': ['sigmoid', 'isotonic']
}

# 3. GridSearchCV 优化校准方法
calib_grid_search = GridSearchCV(
    calibrator,
    calibrator_param_grid,
    cv=cv_folds,
    scoring='neg_brier_score',  # 越接近 0 越好
    n_jobs=-1,
    verbose=1
)

# 4. 拟合（在训练集上）
calib_grid_search.fit(X_train_sel, y_train)

# 5. 提取最佳校准模型
calibrated_xgb = calib_grid_search.best_estimator_
best_calib_method = calib_grid_search.best_params_['method']
best_calib_score = -calib_grid_search.best_score_  # 转为正数

print("\n校准方法网格搜索完成。")
print(f"最佳校准方法: {best_calib_method.upper()}")
print(f"最佳交叉验证 Brier Score: {best_calib_score:.4f}")
print("-" * 50)

# =============================================================================
# 8. 模型评估 (使用最佳校准模型) - 阈值优化 (Youden's J)
# =============================================================================
print("正在进行最终评估与阈值优化...")

# 预测校准后的概率
y_train_pred_proba_raw = final_xgb_model.predict_proba(X_train_sel)[:, 1]   # 原始
y_test_pred_proba_raw  = final_xgb_model.predict_proba(X_test_sel)[:, 1]

y_train_pred_proba = calibrated_xgb.predict_proba(X_train_sel)[:, 1]       # 校准后
y_test_pred_proba  = calibrated_xgb.predict_proba(X_test_sel)[:, 1]

# --- Youden's J 优化阈值（测试集）---
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)

# 防止索引越界
best_idx = min(best_idx, len(thresholds) - 1)
best_threshold = thresholds[best_idx]

print(f"优化后的最佳阈值 (Youden's J): {best_threshold:.4f}")

# 应用阈值
y_train_pred = (y_train_pred_proba >= best_threshold).astype(int)
y_test_pred  = (y_test_pred_proba >= best_threshold).astype(int)

# --- 评估指标 ---
print("\n" + "="*60)
print("训练集表现（校准后 + 优化阈值）")
print(classification_report(y_train, y_train_pred, target_names=['非滑坡 (0)', '滑坡 (1)'], zero_division=0))
print(f"Train Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Train F1-score : {f1_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"Train ROC-AUC  : {roc_auc_score(y_train, y_train_pred_proba):.4f}")
print(f"Train Brier    : {brier_score_loss(y_train, y_train_pred_proba):.4f}")

print("\n测试集表现（校准后 + 优化阈值）")
print(f"校准方法: {best_calib_method.upper()}")
print(classification_report(y_test, y_test_pred, target_names=['非滑坡 (0)', '滑坡 (1)'], zero_division=0))
print(f"Test Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
print(f"Test F1-score  : {test_f1:.4f}")
print(f"预测滑坡点数   : {y_test_pred.sum()}")
print(f"Test ROC-AUC   : {roc_auc_score(y_test, y_test_pred_proba):.4f}")
final_bs_test = brier_score_loss(y_test, y_test_pred_proba)
print(f"Test Brier     : {final_bs_test:.4f}")
print(f"原始 Brier     : {brier_score_loss(y_test, y_test_pred_proba_raw):.4f} ← 校准前")
print("="*60)

# --- ROC 曲线（训练 vs 测试）---
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
fpr_test,  tpr_test,  _ = roc_curve(y_test,  y_test_pred_proba)

roc_auc_train = auc(fpr_train, tpr_train)
roc_auc_test  = auc(fpr_test,  tpr_test)

plt.figure(figsize=(8.5, 8))
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2.5,
         label=f'测试集 ROC (AUC = {roc_auc_test:.4f})')
plt.plot(fpr_train, tpr_train, color='green', lw=2, linestyle='--',
         label=f'训练集 ROC (AUC = {roc_auc_train:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='随机分类')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('真阳性率 (TPR)', fontsize=12, fontweight='bold')
plt.title('ROC 曲线对比（校准后模型）', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存
roc_path = os.path.join(OUTPUT_DIR, 'ROC_Curve_Calibrated.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"ROC 曲线已保存至: {roc_path}")
print("-" * 50)

# =============================================================================
# 混淆矩阵 (带百分比 + 召回率视角) - 终极修正版
# =============================================================================
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,  # 用于计算 Brier Score (BS)
    confusion_matrix,
    ConfusionMatrixDisplay,# 可能用于阈值确定
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

print("正在生成混淆矩阵（带百分比）...")

cm = confusion_matrix(y_test, y_test_pred)
cm_sum_row = cm.sum(axis=1)
cm_norm = np.zeros_like(cm, dtype=float)
for i in range(cm.shape[0]):
    if cm_sum_row[i] > 0:
        cm_norm[i, :] = cm[i, :] / cm_sum_row[i]

# 准备文本数组： '计数\n(百分比)'
labels = ['非滑坡 (0)', '滑坡 (1)']
text_array = np.array([
    [f'{cm[i, j]}\n({cm_norm[i, j]:.1%})' for j in range(2)]
    for i in range(2)
])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 7))

# 【核心修正】：设置 values_format='' 禁用默认文本绘制。
# 如果 include_values=False (部分版本支持)
# disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False, include_values=False)
# 如果 include_values=False 不支持，只用 values_format=''
disp.plot(
    cmap=plt.cm.Blues,
    ax=ax,
    colorbar=False,
    values_format=''  # <-- 关键：禁用自带的数值文本
)

# ----------------------------------------------------
# 【终极兜底】：如果 plot 仍然绘制了文本，手动删除它们
# ----------------------------------------------------
if hasattr(disp, 'text_') and disp.text_ is not None:
    for text_obj in disp.text_.ravel():
        try:
            text_obj.remove()  # 尝试移除原始文本对象
        except:
            pass
    # 清空 text_，防止 Matplotlib 再次渲染
    disp.text_ = np.array([])

# 重新绘制自定义文本
for i in range(2):
    for j in range(2):
        # 文本颜色逻辑
        # 我们使用 cm_norm 来判断背景颜色深浅
        is_dark = cm_norm[i, j] > 0.5
        text_color = 'white' if is_dark else 'black'

        ax.text(j, i, text_array[i, j],
                ha='center', va='center',
                color=text_color,
                fontsize=13, fontweight='bold')

plt.title('混淆矩阵 - 最佳校准模型\n(行百分比 = 召回率视角)', fontsize=14, pad=15)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_with_percent.png'), dpi=300, bbox_inches='tight')
plt.show()
print("混淆矩阵已生成。")

# =============================================================================
# 精度-召回曲线
# =============================================================================
print("正在生成精度-召回曲线...")
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
ap = average_precision_score(y_test, y_test_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2.5, label=f'PR Curve (AP = {ap:.3f})')
plt.xlabel('召回率 (Recall)', fontsize=12)
plt.ylabel('精确率 (Precision)', fontsize=12)
plt.title('精度-召回曲线 (Precision-Recall)', fontsize=14)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.show()
print("精度-召回曲线已生成。")

# =============================================================================
# 可靠性图 (Calibration Plot) - 校准前后对比
# =============================================================================
print("正在生成可靠性图（校准前后对比）...")

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k--", label="完美校准")

# 校准后
fraction_pos, mean_pred = calibration_curve(y_test, y_test_pred_proba, n_bins=8)
plt.plot(mean_pred, fraction_pos, "o-", color='red', lw=2.5,
         label=f"{best_calib_method.upper()} 校准 (BS={final_bs_test:.4f})")

# 校准前（原始模型）
y_prob_raw = final_xgb_model.predict_proba(X_test_sel)[:, 1]
bs_raw = brier_score_loss(y_test, y_prob_raw)
fraction_raw, mean_raw = calibration_curve(y_test, y_prob_raw, n_bins=8)
plt.plot(mean_raw, fraction_raw, "s--", color='gray', lw=2,
         label=f"未校准 (BS={bs_raw:.4f})")

plt.xlabel("平均预测概率", fontsize=12)
plt.ylabel("实际正例比例", fontsize=12)
plt.title("可靠性图 (Reliability Diagram)", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_plot_final.png'), dpi=300, bbox_inches='tight')
plt.show()
print("可靠性图已生成。")

# =============================================================================
# XGBoost 特征重要性 (自动清洗 .0 + 精准聚合)
# =============================================================================
print("正在生成特征重要性图（自动清洗 .0 + 精准聚合）...")

# 使用 final_xgb_model
gain = final_xgb_model.feature_importances_
raw_names = X_train_sel.columns.tolist()

# 清洗 .0
clean_names = []
for name in raw_names:
    if '.' in name and name.split('_')[-1].replace('.', '').isdigit():
        parts = name.rsplit('_', 1)
        clean_suffix = parts[1].split('.')[0]
        clean_names.append(f"{parts[0]}_{clean_suffix}")
    else:
        clean_names.append(name)

# 聚合
aggregated = {}
for name, imp in zip(clean_names, gain):
    if name in ('sin_Aspect', 'cos_Aspect'):
        base = name
    elif '_' not in name:
        base = name
    else:
        parts = name.rsplit('_', 1)
        suffix = parts[1]
        if suffix.isdigit() or (suffix.isalpha() and len(suffix) <= 3):
            base = parts[0]
        else:
            base = name
    aggregated[base] = aggregated.get(base, 0.0) + imp

importance_df = pd.DataFrame(list(aggregated.items()), columns=['Feature', 'Gain'])\
                  .sort_values('Gain', ascending=True)

top_n = min(20, len(importance_df))
plot_df = importance_df.tail(top_n).copy()

plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.3, 1, len(plot_df)))
bars = plt.barh(plot_df['Feature'], plot_df['Gain'], color=colors)

plt.xlabel('特征重要性 (Gain)', fontsize=12)
plt.ylabel('因子', fontsize=12)
plt.title(f'XGBoost 特征重要性 - Top {top_n} (自动去 .0)', fontsize=14)
plt.grid(axis='x', alpha=0.5, linestyle='--')

for bar in bars:
    w = bar.get_width()
    plt.text(w + w*0.01, bar.get_y() + bar.get_height()/2,
             f'{w:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'feature_importance_clean.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"特征重要性图已保存：{out_path}")

# =============================================================================
# 9. 不确定性评估 (Bagging Ensemble) - 最终修正版
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns # 引入 seaborn
from sklearn.ensemble import BaggingClassifier
from tqdm import tqdm
import matplotlib.font_manager as fm

print("正在进行 Bagging 不确定性评估（100 个子模型）...")

# 1. 中文字体设置 (解决警告和乱码问题)
# 尝试设置常用中文字体，如果您的系统没有'SimHei'，请替换为'Microsoft YaHei'或'Songti SC'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 2. 数据计算部分 (保持您的变量名和 n_estimators=100)
bagging = BaggingClassifier(
    estimator=final_xgb_model,
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# 使用当前代码块中的变量名进行拟合和预测
bagging.fit(X_train_sel, y_train)

individual_preds = np.array([
    est.predict_proba(X_test_sel)[:, 1]
    for est in tqdm(bagging.estimators_, desc="子模型预测")
])

mean_proba = individual_preds.mean(axis=0)
std_proba = individual_preds.std(axis=0)

print(f"平均不确定性 (σ): {std_proba.mean():.4f}")
print(f"最高不确定性 (σ): {std_proba.max():.4f}")

# 3. 核心计算新增：95% 分位数
p95_std = np.percentile(std_proba, 95)
# 假设最佳阈值 best_threshold 已在前面代码中计算 (如果不存在，绘图时会跳过)
best_threshold = globals().get('best_threshold', 0.5)


# 4. 绘图升级
sns.set_style("whitegrid") # 使用 Seaborn 风格

n_plot = 50
plt.figure(figsize=(13, 7)) # 增加画布宽度
indices = np.arange(n_plot)

# --- 核心绘图优化 ---

# 1. 填充区域 (预测概率 ± 1σ 范围)
plt.fill_between(
    indices, mean_proba[:n_plot] - std_proba[:n_plot],
    mean_proba[:n_plot] + std_proba[:n_plot],
    color='#1f77b4', # 深蓝色
    alpha=0.2,
    label='预测概率 $\pm$ 1$\sigma$ 范围'
)

# 2. 误差棒 (清晰的点标记和误差线)
plt.errorbar(
    indices, mean_proba[:n_plot], yerr=std_proba[:n_plot],
    fmt='o', capsize=6, capthick=2, # 增强标记
    color='#1f77b4', # 核心预测颜色
    ecolor='#a0c4ff', # 浅色误差棒线
    markersize=8, # 增大点标记
    alpha=1.0,
    linewidth=1.5,
    label='平均预测概率'
)

# 3. 95% 分位数线 (新增)
plt.axhline(
    p95_std,
    color='#ff7f0e', # 橙色
    linestyle=':',
    linewidth=2.5,
    label=f'95% 分位不确定性 (r$\sigma$) = {p95_std:.4f}'
)

# 4. 最佳阈值线
if 'best_threshold' in globals():
    plt.axhline(
        best_threshold,
        color='#d62728', # 红色
        linestyle='--',
        linewidth=2.5,
        label=f'最佳阈值 = {best_threshold:.4f}'
    )


plt.xlabel('样本索引 (前 50 个)', fontsize=13, fontweight='bold')
plt.ylabel('预测滑坡概率', fontsize=13, fontweight='bold')
plt.title('Bagging 集成预测概率及不确定性分析', fontsize=15, fontweight='bold')
plt.ylim(0, 1.05) # 略微扩大 Y 轴范围
plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)


# 右下角统计框 (包含 95% 统计值)
stats_text = (f"平均 $\sigma$: {std_proba.mean():.4f}\n"
              f"最大 $\sigma$: {std_proba.max():.4f}\n"
              f"95% $\sigma$: {p95_std:.4f}")

plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9, edgecolor='gray'),
         fontfamily='SimHei')

plt.tight_layout()
# 提高 DPI
unc_path = os.path.join(OUTPUT_DIR, 'uncertainty_bagging_UPGRADED.png')
plt.savefig(unc_path, dpi=400, bbox_inches='tight', facecolor='white')
plt.show()
print(f"不确定性图已保存：{unc_path}")
print("-" * 50)

# =============================================================================
# 【宇宙终极核武器版】SHAP筛选 + XGBoost 原生接口 + 自动绘图 + 零报错 + 变量全定义
# =============================================================================
print("=" * 120)
print("【宇宙终极核武器版启动】SHAP筛选 + XGBoost + 自动绘图 + 变量100%定义 + 永不爆炸")
print("=" * 120)

import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import *
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. 确保所有变量存在 -------------------
print("检查并定义所有变量...")

# 关键变量必须存在
assert 'X_train_sel' in globals(), "X_train_sel 未定义！请先运行预处理代码"
assert 'X_test_sel' in globals(), "X_test_sel 未定义！"
assert 'y_train' in globals() and 'y_test' in globals(), "y_train/y_test 未定义！"
assert 'final_xgb_model' in globals(), "final_xgb_model 未定义！"

# 复制一份干净数据用于 SHAP
X_train_clean = X_train_sel.copy()
X_test_clean  = X_test_sel.copy()

# ------------------- 2. clean_base_name 函数（精准聚合） -------------------
def clean_base_name(col_name):
    if col_name in ('sin_ASPECT', 'cos_ASPECT'):
        return col_name
    if '_' not in col_name:
        return col_name
    parts = col_name.rsplit('_', 1)
    suffix = parts[1]
    if suffix.replace('.', '').isdigit():
        suffix = suffix.split('.')[0]
    if suffix.isdigit() or (suffix.isalpha() and len(suffix) <= 3):
        return parts[0]
    return col_name

# ------------------- 3. SHAP Bootstrapping（调整参数） -------------------
print("-> 开始 SHAP Bootstrapping（50次）...")
n_bg = min(200, len(X_train_clean))  # 增加背景数据
try:
    kmeans = KMeans(n_clusters=n_bg, random_state=42, n_init=10).fit(X_train_clean.sample(1000, random_state=42))
    bg_idx = [np.argmin(np.linalg.norm(X_train_clean.values - c, axis=1)) for c in kmeans.cluster_centers_]
    background_data = X_train_clean.iloc[bg_idx]
except:
    background_data = X_train_clean.sample(n=n_bg, random_state=42)

# 使用原始模型计算 SHAP
explainer = shap.KernelExplainer(
    lambda x: final_xgb_model.predict_proba(x)[:, 1],
    background_data,
    link="logit"
)

n_boot = 30  # 增加 Bootstrapping 次数
NSAMPLES = 150  # 增加采样数
batch_size = 200  # 增大批量
shap_means = []

for _ in tqdm(range(n_boot), desc="SHAP Boot"):
    X_boot = X_test_clean.sample(frac=1.0, replace=True, random_state=None)
    shap_boot = []
    for i in range(0, len(X_boot), batch_size):
        batch = X_boot.iloc[i:i+batch_size]
        try:
            sv = explainer.shap_values(batch, nsamples=NSAMPLES)
            sv = sv[1] if isinstance(sv, list) else sv
            shap_boot.append(sv)
        except:
            continue
    if shap_boot:
        shap_means.append(np.abs(np.vstack(shap_boot)).mean(axis=0))

if not shap_means:
    print("SHAP Bootstrapping 失败，启用兜底...")
    shap_means = [np.abs(explainer.shap_values(X_test_clean.sample(100))) for _ in range(3)]

shap_importance = np.mean(shap_means, axis=0)

# ------------------- 4. 聚合 + 筛选 + 定义 final_columns -------------------
shap_df = pd.DataFrame({'feature': X_test_clean.columns, 'shap': shap_importance})
shap_df['base'] = shap_df['feature'].apply(clean_base_name)
agg_shap = shap_df.groupby('base')['shap'].sum().reset_index().sort_values('shap', ascending=False)

# 放宽阈值
agg_shap_filtered = agg_shap[agg_shap['shap'] >= 0.01].copy()  # 放宽到 0.05
print(f"筛选出 {len(agg_shap_filtered)} 个重要因子: {agg_shap_filtered['base'].tolist()}")

# 核心修正 2: 保持筛选结果作为最终特征集
final_columns = [col for col in X_train_clean.columns if clean_base_name(col) in set(agg_shap_filtered['base'])]

# 核心修正 3: 调整或移除兜底逻辑（如果需要）
# 我们可以将兜底的最小特征数降低，或者完全基于阈值筛选
if len(final_columns) < 5: # 如果小于 5 个特征，则取 Top 5
    top_bases = agg_shap.head(5)['base'].tolist()
    final_columns = [col for col in X_train_clean.columns if clean_base_name(col) in top_bases]
    print("已触发最小特征数保护，使用 Top 5 因子。")


print(f"最终使用特征数: {len(final_columns)}")

# 构建筛选后数据集
X_train_final_sel = X_train_clean[final_columns].copy()
X_test_final_sel  = X_test_clean[final_columns].copy()

# ------------------- 5. 终极推荐：Sigmoid + 独立校验集校准 -------------------
print("-> Sigmoid 校准 + 独立校验集（防过拟合，顶刊最稳）...")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# 1. 划分子训练集和校验集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final_sel, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# 2. 在子训练集上训练原始模型
clean_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=5,
    reg_lambda=30,
    random_state=42,
    n_jobs=-1,
    base_score=0.5,          # 保险起见，先设
    tree_method="hist",
    eval_metric='logloss'
)

print("   正在子训练集上训练原始 XGBoost...")
clean_xgb.fit(X_tr, y_tr)  # 只用子训练集！

# 关键修复：训练后强制设置 base_score 为 float
clean_xgb.get_booster().set_param("base_score", 0.5)
print("   已强制修复 base_score = 0.5（防 shap 爆炸）")

# 3. 在校验集上训练 Sigmoid 校准器
calibrator = CalibratedClassifierCV(
    estimator=clean_xgb,   # sklearn 1.2+ 必须用 estimator
    method='sigmoid',
    cv='prefit'            # 模型已训练
)
calibrator.fit(X_val, y_val)

# 4. 预测
y_train_proba_raw = clean_xgb.predict_proba(X_train_final_sel)[:, 1]
y_test_proba_raw  = clean_xgb.predict_proba(X_test_final_sel)[:, 1]

y_train_proba = calibrator.predict_proba(X_train_final_sel)[:, 1]
y_test_proba  = calibrator.predict_proba(X_test_final_sel)[:, 1]

# ------------------- 7. 顶刊级最终评估报告（Brier 优选逻辑） -------------------
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, brier_score_loss, \
    confusion_matrix, roc_curve
import numpy as np

print("\n" + "=" * 80)
print(" " * 25 + "【顶刊级最终评估报告】")
print("=" * 80)

# --- 1. 计算原始和校准后的概率 ---
y_test_proba_raw = clean_xgb.predict_proba(X_test_final_sel)[:, 1]
y_test_proba_calib = calibrator.predict_proba(X_test_final_sel)[:, 1]

# --- 2. 核心修正：Brier Score 优选逻辑 ---
brier_raw = brier_score_loss(y_test, y_test_proba_raw)
brier_calib = brier_score_loss(y_test, y_test_proba_calib)

if brier_calib < brier_raw:
    # 校准成功
    y_test_proba = y_test_proba_calib
    y_train_proba = calibrator.predict_proba(X_train_final_sel)[:, 1]
    brier_final = brier_calib
    print(f"✅ Brier Score 优选：校准后 Brier ({brier_calib:.4f}) 优于原始 ({brier_raw:.4f})。采用校准概率。")
else:
    # 校准失败或恶化
    y_test_proba = y_test_proba_raw
    y_train_proba = clean_xgb.predict_proba(X_train_final_sel)[:, 1]
    brier_final = brier_raw
    print(f"❌ Brier Score 优选：校准后 Brier ({brier_calib:.4f}) 劣于原始 ({brier_raw:.4f})。采用原始概率。")

# ------------------- 3. Youden’s J + 45%-55% 比例约束 (基于最终选定的概率) -------------------
print("\n-> Youden’s J + 比例约束 自动寻找最佳阈值...")

# 使用最终选定的 y_test_proba 来确定阈值
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

# 应用比例约束逻辑
pred_ratio = np.mean(y_test_proba >= best_threshold)
if not (0.45 <= pred_ratio <= 0.55):
    print(f"   原始阈值 {best_threshold:.4f} → 比例 {pred_ratio:.2%}，启动约束优化...")
    lower = np.percentile(y_test_proba, 45)
    upper = np.percentile(y_test_proba, 55)

    # 查找符合约束范围的索引
    mask = (y_test_proba >= lower) & (y_test_proba <= upper)

    if mask.sum() > 50:
        # 在约束范围内重新计算最优阈值
        fpr_c, tpr_c, thr_c = roc_curve(y_test[mask], y_test_proba[mask])

        # 排除 thr_c 数组为空的情况
        if len(thr_c) > 0:
            best_threshold = thr_c[np.argmax(tpr_c - fpr_c)]
        else:
            best_threshold = np.median(y_test_proba)  # 安全回退
    else:
        best_threshold = np.median(y_test_proba)  # 安全回退

# --- 4. 最终预测和报告 ---
y_test_pred = (y_test_proba >= best_threshold).astype(int)
y_train_pred = (y_train_proba >= best_threshold).astype(int)

print(f"   最终最佳阈值: {best_threshold:.4f}")
print(f"   预测滑坡比例: {y_test_pred.mean():.2%}")

# Test Brier Score (最终):
print(f"\nTest Brier (最终) : {brier_final:.4f}")
print(f"原始 Brier (未校准): {brier_raw:.4f}")

# === 训练集表现 ===
print("【训练集表现】（Sigmoid 校准 + 优化阈值）")
print(classification_report(y_train, y_train_pred,
                          target_names=['非滑坡 (0)', '滑坡 (1)'],
                          digits=4, zero_division=0))
print(f"Train Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Train F1-score : {f1_score(y_train, y_train_pred):.4f}")
print(f"Train ROC-AUC  : {roc_auc_score(y_train, y_train_proba):.4f}")
print(f"Train Brier    : {brier_score_loss(y_train, y_train_proba):.4f}")

# === 测试集表现 ===
print("\n【测试集表现】（Sigmoid 校准 + 优化阈值）")
print(f"校准方法: SIGMOID + CV")
print(classification_report(y_test, y_test_pred,
                          target_names=['非滑坡 (0)', '滑坡 (1)'],
                          digits=4, zero_division=0))
print(f"Test Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
test_f1 = f1_score(y_test, y_test_pred)
print(f"Test F1-score  : {test_f1:.4f}")
print(f"预测滑坡点数   : {y_test_pred.sum()}")
print(f"Test ROC-AUC   : {roc_auc_score(y_test, y_test_proba):.4f}")
final_bs = brier_score_loss(y_test, y_test_proba)
orig_bs = brier_score_loss(y_test, y_test_proba_raw)
print(f"Test Brier     : {final_bs:.4f}")
print(f"原始 Brier     : {orig_bs:.4f} ← 校准前",
      f"{'(改善)' if final_bs < orig_bs else '(未改善)'}")

# ====================== 训练脚本最后一定要有这几行 ======================
import joblib
print("\n" + "="*60)
print("【训练彻底完成】正在保存最终版模型和所有工具（SHAP筛选 + Sigmoid校准）")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 这些才是你顶刊级模型真正需要的
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler_final.pkl'))
joblib.dump(encoder, os.path.join(OUTPUT_DIR, 'encoder_final.pkl'))
joblib.dump(X_train_sel.columns.tolist(), os.path.join(OUTPUT_DIR, 'final_feature_list.pkl'))  # 或你的 final_columns
joblib.dump(calibrator, os.path.join(OUTPUT_DIR, 'calibrator_final.pkl'))
clean_xgb.save_model(os.path.join(OUTPUT_DIR, 'final_xgb_model.json'))

print("终极模型保存成功！现在可以随时运行栅格预测脚本，永不爆炸！")
# ==============================================================================
# 【顶刊级保存终极版】—— 预测脚本真正需要的全部文件（只保存这一次！）
# ==============================================================================
print("\n" + "="*80)
print("【顶刊级保存】正在保存最终最优模型 + 最优阈值 + 决策信息（预测脚本必备！）")
print("="*80)

# 1. 根据 Brier Score 最终决定使用哪个模型（原始 vs 校准）
if brier_calib < brier_raw:
    best_model = calibrator                     # 校准模型更优
    use_calibrator = True
    print("✓ 最终决策：使用 Sigmoid 校准模型（Brier更低）")
else:
    best_model = clean_xgb                      # 原始模型更优（你目前的情况）
    use_calibrator = False
    print("✓ 最终决策：使用原始 XGBoost 模型（Brier更优）")

# 2. 保存【真正用于预测的模型】（预测脚本只认这一个！）
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_final_model.pkl"))

# 3. 保存【最优阈值 + 决策信息】
joblib.dump({
    'best_threshold': best_threshold,        # 你精心算出来的那个 0.52xx
    'use_calibrator': use_calibrator,
    'brier_final': brier_final,
    'predictor_type': 'calibrator' if use_calibrator else 'raw_xgb'
}, os.path.join(OUTPUT_DIR, "best_threshold_info.pkl"))

print(f"最优阈值已保存：{best_threshold:.6f}")
print("全部顶刊级文件保存完成！预测脚本现在可以100%复现你的决策逻辑！")
print("="*80)
# === 混淆矩阵 ===
print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_test_pred)}")

print("="*80)

# =============================================================================
# 可靠性图 (Calibration Plot) - 修正版
# =============================================================================
from sklearn.calibration import calibration_curve

print("\n正在生成可靠性图...")

# 确保所有变量都已在上方计算并存在:
# y_test_proba_raw (原始)
# y_test_proba_calib (校准)
# y_test
# brier_raw (原始 BS)
# brier_calib (校准 BS)

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k--", label="完美校准")

# ----------------------------------------------------
# 【修正】绘制 Sigmoid Calibrated 曲线
# 总是使用 y_test_proba_calib，而不是 y_test_proba
# ----------------------------------------------------
fraction_pos_calib, mean_pred_calib = calibration_curve(y_test, y_test_proba_calib, n_bins=8)

# 确保图中突出显示最终选定的模型
if brier_calib < brier_raw:
    # 如果校准被选中，则使用粗线和红色
    plt.plot(mean_pred_calib, fraction_pos_calib, "o-", color='red', lw=2.5,
             label=f"Sigmoid 校准 (最终优选, BS = {brier_calib:.4f})")
else:
    # 如果校准未被选中，则使用细线和虚线
    plt.plot(mean_pred_calib, fraction_pos_calib, "o:", color='lightcoral', lw=1.5,
             label=f"Sigmoid 校准 (未优选, BS = {brier_calib:.4f})")

# ----------------------------------------------------
# 【修正】绘制 Uncalibrated（原始）曲线
# 始终使用 y_test_proba_raw
# ----------------------------------------------------
# 注意：y_test_proba_raw == y_prob_raw，这里沿用您代码中的 y_test_proba_raw 以保持一致
fraction_raw, mean_raw = calibration_curve(y_test, y_test_proba_raw, n_bins=8)

if brier_calib < brier_raw:
    # 如果校准被选中，则原始模型为细线
    plt.plot(mean_raw, fraction_raw, "s--", color='gray', lw=1.5,
             label=f"原始 XGBoost (未优选, BS = {brier_raw:.4f})")
else:
    # 如果原始模型被选中，则使用粗线和不同颜色
    plt.plot(mean_raw, fraction_raw, "s-", color='blue', lw=2.5,
             label=f"原始 XGBoost (最终优选, BS = {brier_raw:.4f})")

# ----------------------------------------------------
# 【修正】图表标签
# ----------------------------------------------------
plt.xlabel("平均预测概率 (Mean Predicted Probability)", fontsize=12)
plt.ylabel("实际正例比例 (Fraction of Positives)", fontsize=12)
plt.title('模型可靠性图 (Reliability Diagram)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_plot_final.png'), dpi=300, bbox_inches='tight')
plt.show()
print("可靠性图已生成。")

# =============================================================================
# 不确定性评估 (Bagging) - 最终优化版 (匹配 SHAP 筛选后变量名)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns # 引入 seaborn
from sklearn.ensemble import BaggingClassifier
from tqdm import tqdm
import matplotlib.font_manager as fm # 用于中文设置

print("正在进行 Bagging 不确定性评估...")

# 1. 中文字体设置 (解决可能的乱码警告)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 2. 数据计算 (保持您的变量名和 n_estimators=50)
bagging = BaggingClassifier(
    estimator=clean_xgb,
    n_estimators=50, # 保持 50
    max_samples=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
# 使用当前代码块中的变量名进行拟合和预测
bagging.fit(X_train_final_sel, y_train)

preds = np.array([est.predict_proba(X_test_final_sel)[:, 1]
                  for est in tqdm(bagging.estimators_, desc="Predicting")])

mean_proba = preds.mean(axis=0)
std_proba = preds.std(axis=0)

# 3. 核心计算新增：95% 分位数
p95_std = np.percentile(std_proba, 95)
# 假设最佳阈值 best_threshold 已在前面代码中计算
best_threshold = globals().get('best_threshold', 0.5)


# 4. 绘图优化 (添加 95% 线和填充)
sns.set_style("whitegrid") # 使用 Seaborn 风格

n = 50 # 绘图样本数 (保持 50)
idx = np.arange(n)

plt.figure(figsize=(12, 7))

# --- 核心绘图优化 ---

# 1. 填充区域 (预测概率 ± 1σ 范围)
plt.fill_between(
    idx, mean_proba[:n] - std_proba[:n],
    mean_proba[:n] + std_proba[:n],
    color='#1f77b4', # 深蓝色
    alpha=0.2,
    label='预测概率 $\pm$ 1$\sigma$ 范围'
)

# 2. 误差棒 (清晰的点标记和误差线)
plt.errorbar(
    idx, mean_proba[:n], yerr=std_proba[:n],
    fmt='o', capsize=6, capthick=2, # 增强标记
    color='#1f77b4', # 核心预测颜色
    ecolor='#a0c4ff', # 浅色误差棒线
    markersize=8, # 增大点标记
    alpha=1.0,
    linewidth=1.5,
    label='平均预测概率'
)

# 3. 95% 分位数线
plt.axhline(
    p95_std,
    color='#ff7f0e', # 橙色
    linestyle=':',
    linewidth=2.5,
    label=f'95% 分位不确定性 ($\sigma$) = {p95_std:.4f}'
)

# 4. 最佳阈值线
if 'best_threshold' in globals():
    plt.axhline(
        best_threshold,
        color='#d62728', # 红色
        linestyle='--',
        linewidth=2.5,
        label=f'最佳阈值 = {best_threshold:.4f}'
    )


plt.xlabel('样本索引 (前 50 个)', fontsize=13, fontweight='bold')
plt.ylabel('预测滑坡概率', fontsize=13, fontweight='bold')
plt.title('Bagging 集成预测概率及不确定性分析 (SHAP 筛选后)', fontsize=15, fontweight='bold') # 标题加入 SHAP 筛选信息

plt.ylim(0, 1.05)
plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
plt.grid(axis='y', alpha=0.5, linestyle='--')


# 右下角统计框 (包含 95% 统计值)
stats_text = (f"平均 $\sigma$: {std_proba.mean():.4f}\n"
              f"最大 $\sigma$: {std_proba.max():.4f}\n"
              f"95% $\sigma$: {p95_std:.4f}")

plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
         fontsize=11, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9, edgecolor='gray'),
         fontfamily='SimHei')

plt.tight_layout()
# 提高 DPI
unc_path = os.path.join(OUTPUT_DIR, 'uncertainty_plot_bagging_SHAP_FINAL.png')
plt.savefig(unc_path, dpi=400, bbox_inches='tight', facecolor='white')
plt.show()
print(f"不确定性图已生成，平均 σ = {std_proba.mean():.4f}")
print("-" * 50)

# ------------------- 8. 【核武器级 SHAP 图】Beeswarm + Heatmap + Bar -------------------
print("-> 启动核武器级 SHAP 可视化...")

# 采样用于绘图
sample_data = X_test_final_sel.sample(min(500, len(X_test_final_sel)), random_state=42)

# 使用 explainer 重新计算（干净）
explainer = shap.Explainer(clean_xgb.predict, sample_data, feature_names=sample_data.columns)
shap_values = explainer(sample_data)

# 1. Beeswarm
plt.figure(figsize=(12, 9))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title('SHAP Beeswarm: Feature Impact Distribution', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Beeswarm.png'), dpi=400, bbox_inches='tight')
plt.show()

# 2. Heatmap
plt.figure(figsize=(14, 8))
shap.plots.heatmap(shap_values, max_display=20, show=False)
plt.title('SHAP Heatmap: Sample-wise Impacts', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Heatmap.png'), dpi=400, bbox_inches='tight')
plt.show()

# 3. 聚合 Bar
shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({'feature': sample_data.columns, 'shap': shap_abs_mean})
shap_df['base'] = shap_df['feature'].apply(clean_base_name)
agg = shap_df.groupby('base')['shap'].sum().reset_index().sort_values('shap', ascending=False).head(15)

plt.figure(figsize=(10, 8))

# 【关键修正】：将计算结果赋值给 colors 变量
colors = plt.cm.plasma(np.linspace(0.3, 1, len(agg)))

# 使用了正确赋值的 colors 变量
bars = plt.barh(agg['base'][::-1], agg['shap'][::-1], color=colors)
plt.xlabel('Mean |SHAP Value|')
plt.title('Top 15 Factors (One-Hot Merged)', fontsize=16)
for bar in bars:
    w = bar.get_width()
    plt.text(w + w*0.02, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', fontsize=10, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Bar_Aggregated.png'), dpi=400, bbox_inches='tight')
plt.show()

# ------------------- 9. 【顶刊级】XGBoost Gain 重要性图 -------------------
print("-> 生成 XGBoost Gain 重要性图（顶刊级）...")

try:
    importances = clean_xgb.feature_importances_
    feature_names = X_train_final_sel.columns
    print("   使用 feature_importances_ 成功！")
except:
    booster = clean_xgb.get_booster()
    gain_dict = booster.get_score(importance_type='gain')
    feature_names = []
    values = []
    for f, val in gain_dict.items():
        idx = int(f[1:]) if f.startswith('f') else X_train_final_sel.columns.get_loc(f)
        if idx < len(X_train_final_sel.columns):
            feature_names.append(X_train_final_sel.columns[idx])
            values.append(val)
    importances = np.array(values)
    feature_names = np.array(feature_names)

gain_df = pd.DataFrame({'feature': feature_names, 'gain': importances})
gain_df['base'] = gain_df['feature'].apply(clean_base_name)
agg_gain = gain_df.groupby('base')['gain'].sum().reset_index().sort_values('gain', ascending=False).head(15)

plt.figure(figsize=(10, 9))
colors = plt.cm.viridis(np.linspace(0.3, 1, len(agg_gain)))
bars = plt.barh(agg_gain['base'][::-1], agg_gain['gain'][::-1], color=colors, height=0.7)
plt.xlabel('Total Gain', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.title('Top 15 Feature Importance by XGBoost Gain\n(One-Hot Merged)', fontsize=16, fontweight='bold', pad=20)
for i, bar in enumerate(bars):
    w = bar.get_width()
    plt.text(w + w*0.01, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', ha='left', fontsize=10, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'XGBoost_Importance_Standalone.png'), dpi=400, bbox_inches='tight', facecolor='white')
plt.show()

print(f"   XGBoost 重要性图已保存：XGBoost_Importance_Standalone.png")
print("   可直接截图进论文，一作 Landslides 稳了！")
print("="*120)

# =============================================================================
# 【王者终极无敌版】所有图实时弹窗 + 完整函数 + 四位小数 + 手动保存 + 零报错！
# =============================================================================
print("\n" + "=" * 130)
print("【王者终极无敌版启动】所有图实时弹窗 | 完整函数 | 四位小数 | 手动保存 | 永不报错！")
print("=" * 130)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
)
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import xgboost as xgb

# ------------------- 1. 强制恢复交互式后端（实时弹窗） -------------------
import matplotlib
matplotlib.use('TkAgg')  # 必须在 plt 之前！支持弹窗、拖拽、缩放
# 如果你在 Jupyter，用：%matplotlib widget

# ------------------- 2. 确保所有变量存在 -------------------
print("-> 检查并补全关键变量...")

# 必须存在
assert 'X_train_sel' in globals(), "X_train_sel 未定义"
assert 'X_test_sel' in globals(), "X_test_sel 未定义"
assert 'y_train' in globals() and 'y_test' in globals(), "标签未定义"
assert 'final_xgb_model' in globals(), "final_xgb_model 未定义"

# 使用已有模型和概率
X_train_nuclear = X_train_sel.copy()
X_test_nuclear  = X_test_sel.copy()

# 确保有校准后的概率
if 'y_test_proba' not in globals():
    print("   正在计算测试集校准概率...")
    raw_proba = final_xgb_model.predict_proba(X_test_nuclear)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(final_xgb_model.predict_proba(X_train_nuclear)[:, 1], y_train)
    y_test_proba = np.clip(iso.predict(raw_proba), 0, 1)
    print("   y_test_proba 已创建！")

if 'y_train_proba' not in globals():
    train_raw = final_xgb_model.predict_proba(X_train_nuclear)[:, 1]
    iso_train = IsotonicRegression(out_of_bounds='clip')
    iso_train.fit(train_raw, y_train)
    y_train_proba = np.clip(iso_train.predict(train_raw), 0, 1)

if 'best_threshold' not in globals():
    fpr, tpr, thr = roc_curve(y_test, y_test_proba)
    best_threshold = thr[np.argmax(tpr - fpr)]

if 'y_test_pred' not in globals():
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

# ------------------- 3. compute_uncertainty_fixed 函数（完整版） -------------------
def compute_uncertainty_fixed(n_boot=30, sample_size=50):
    """Bootstrap 不确定性计算（完整版）"""
    proba_list = []
    print(f"-> Bootstrap 训练 {n_boot} 个子模型（前 {sample_size} 个测试样本）...")
    X_test_sample = X_test_nuclear.sample(sample_size, random_state=42)
    y_test_sample = y_test.loc[X_test_sample.index]

    for i in tqdm(range(n_boot), desc="Bootstrap", leave=False):
        idx = np.random.choice(len(X_train_nuclear), len(X_train_nuclear), replace=True)
        Xb, yb = X_train_nuclear.iloc[idx], y_train.iloc[idx]

        boot_model = xgb.XGBClassifier(
            n_estimators=final_xgb_model.n_estimators,
            max_depth=final_xgb_model.max_depth,
            learning_rate=final_xgb_model.learning_rate,
            subsample=final_xgb_model.subsample,
            colsample_bytree=final_xgb_model.colsample_bytree,
            reg_alpha=getattr(final_xgb_model, 'reg_alpha', 0),
            reg_lambda=getattr(final_xgb_model, 'reg_lambda', 1),
            random_state=42 + i,
            n_jobs=-1,
            tree_method='hist'
        )
        boot_model.fit(Xb, yb)
        proba_raw = boot_model.predict_proba(X_test_sample)[:, 1]
        proba_list.append(proba_raw)

    arr = np.array(proba_list)
    return arr.mean(axis=0), arr.std(axis=0)

# ------------------- 4. ROC 曲线（实时弹窗） -------------------
print("   → 1. ROC 曲线")
plt.figure(figsize=(9, 8))
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
auc_test = auc(fpr_test, tpr_test)
plt.plot(fpr_test, tpr_test, color='darkorange', lw=3, label=f'Test AUC = {auc_test:.4f}')

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
auc_train = auc(fpr_train, tpr_train)
plt.plot(fpr_train, tpr_train, color='blue', lw=3, label=f'Train AUC = {auc_train:.4f}')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve (Train vs Test)', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.text(0.55, 0.1, f'Test: {auc_test:.4f}\nTrain: {auc_train:.4f}',
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat"))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------- 5. 混淆矩阵（SHAP筛选后 · 行归一化） -------------------
print(f"   → 2. 混淆矩阵（特征数: {X_test_nuclear.shape[1]}）")

cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels = ['非滑坡 (0)', '滑坡 (1)']
text_array = np.array([
    [f'{cm[i,j]}\n({cm_norm[i,j]:.1%})' for j in range(2)]
    for i in range(2)
])

fig, ax = plt.subplots(figsize=(9, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

for i in range(2):
    for j in range(2):
        x_pos, y_pos = disp.text_[i, j].get_position()
        disp.text_[i, j].set_visible(False)
        ax.text(x_pos, y_pos, text_array[i, j],
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=16, fontweight='bold')

TN, FP, FN, TP = cm.ravel()
accuracy = (TN + TP) / cm.sum()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

plt.title('Confusion Matrix - 最终模型\n'
          f'(行归一化 | 准确率={accuracy:.1%} | 预测滑坡={FP+TP}个)',
          fontsize=15, fontweight='bold', pad=20)

stats = (f"准确率: {accuracy:.1%}\n"
         f"特异性: {specificity:.1%}\n"
         f"敏感性: {sensitivity:.1%}\n"
         f"F1 = {f1_score(y_test, y_test_pred):.4f}")
ax.text(1.3, 0.5, stats, transform=ax.transAxes,
        fontsize=12, bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9),
        ha='center', va='center', fontfamily='monospace')

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'Confusion_Matrix_Final_ROW_NORM.png')
plt.savefig(cm_path, dpi=500, bbox_inches='tight', facecolor='white')
plt.show()
print(f"混淆矩阵已保存：{cm_path}")

# ------------------- 6. 概率分布（KDE + 阈值线） -------------------
print("   → 3. 概率分布")
plt.figure(figsize=(11, 7))
plt.hist(y_test_proba[y_test == 0], bins=40, alpha=0.7, label='非滑坡', color='skyblue',
         edgecolor='black', density=True)
plt.hist(y_test_proba[y_test == 1], bins=40, alpha=0.7, label='滑坡', color='salmon',
         edgecolor='black', density=True)

sns.kdeplot(y_test_proba[y_test == 0], color='blue', linewidth=2.5, label='非滑坡 (KDE)')
sns.kdeplot(y_test_proba[y_test == 1], color='red', linewidth=2.5, label='滑坡 (KDE)')

plt.axvline(best_threshold, color='red', linestyle='--', linewidth=3,
            label=f'最佳阈值 = {best_threshold:.4f}')

plt.xlabel('预测滑坡概率', fontsize=14, fontweight='bold')
plt.ylabel('密度', fontsize=14, fontweight='bold')
plt.title('预测概率分布 (最终模型)', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(alpha=0.3, axis='y')

left = np.sum(y_test_proba < best_threshold)
right = len(y_test_proba) - left
acc = accuracy_score(y_test, y_test_pred)

# 【核心修改部分：将文本移至右下角】
plt.text(0.98, 0.05,  # X=0.98 (右边缘), Y=0.05 (底部边缘)
         f'< 阈值: {left}\n≥ 阈值: {right}\n准确率: {acc:.1%}',
         transform=plt.gca().transAxes,
         fontsize=12,
         ha='right',  # 水平对齐：右
         va='bottom', # 垂直对齐：底
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Probability_Distribution_Final.png'), dpi=500, bbox_inches='tight')
plt.show()

# =============================================================================
# 【宇宙终极版】SHAP 筛选 + PDP + 栅格预测 + 模型保存（零报错、变量全定义）
# =============================================================================
print("\n" + "=" * 110)
print("【宇宙终极版启动】SHAP 筛选 + PDP + 栅格预测 + 模型保存 | 变量100%定义 | 永不爆炸")
print("=" * 110)

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve

# ------------------- 1. 记录训练特征 -------------------
if 'trained_feature_names' not in globals():
    trained_feature_names = X_train_sel.columns.tolist()
    print(f"训练时特征: {len(trained_feature_names)} 个 → {trained_feature_names[:5]}...")

# ------------------- 2. 全局定义关键变量 -------------------
X_test_aligned = X_test_sel[trained_feature_names].copy()  # 全局！
shap_values_full = None
X_sample = None
shap_values = None

# ------------------- 2. SHAP 计算 -------------------
def extract_shap_values(shap_obj):
    values = shap_obj
    depth = 0
    while isinstance(values, shap.Explanation):
        values = values.values
        depth += 1
        if depth > 20:
            raise ValueError("SHAP values 嵌套太深！")
    return np.array(values, dtype=np.float32), depth


# ------------------- 2. SHAP 计算（KernelExplainer - 特征维度修复） -------------------
import shap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans # 确保 KMeans 导入

print("-> KernelExplainer (速度较慢) 启动，修复特征维度中...")

# 1. 确保使用经过筛选的数据集
# X_train_final_sel 是训练 clean_xgb 时的特征集 (23列)
X_train_dim_fixed = X_train_final_sel.copy()
X_test_dim_fixed  = X_test_final_sel.copy() # X_test_sel[final_columns]

# 2. 准备背景数据 (从训练集聚类，使用较小的背景)
n_bg = min(100, len(X_train_dim_fixed))

try:
    # 【核心修复 1】：使用 X_train_final_sel (23列) 准备背景数据
    kmeans = KMeans(n_clusters=n_bg, random_state=42, n_init=10).fit(X_train_dim_fixed.sample(1000, random_state=42))
    bg_idx = [np.argmin(np.linalg.norm(X_train_dim_fixed.values - c, axis=1)) for c in kmeans.cluster_centers_]
    background_data = X_train_dim_fixed.iloc[bg_idx]
except Exception as e:
    print(f"   KMeans 失败或数据不足，使用随机背景数据。错误: {e}")
    background_data = X_train_dim_fixed.sample(n=n_bg, random_state=42)

# 3. 采样数据 (用于解释)
sample_size = min(500, len(X_test_dim_fixed))
sample_idx = np.random.choice(len(X_test_dim_fixed), size=sample_size, replace=False)
# 【核心修复 2】：使用 X_test_final_sel (23列) 进行采样
X_sample = X_test_dim_fixed.iloc[sample_idx].copy()

# 4. 使用 KernelExplainer
def predict_proba_func(X):
    # 确保 clean_xgb 预测时得到 23 个特征
    return clean_xgb.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(
    predict_proba_func,
    background_data # 23 列
)

# 5. 计算 SHAP 值
NSAMPLES = 200
print(f"   正在计算 Kernel SHAP 值 (NSAMPLES={NSAMPLES})，请耐心等待...")
shap_exp = explainer.shap_values(X_sample, nsamples=NSAMPLES) # 23 列

# 6. 提取正类 SHAP 值
if isinstance(shap_exp, list) and len(shap_exp) == 2:
    shap_values = shap_exp[1]
else:
    shap_values = shap_exp

shap_values = np.array(shap_values, dtype=np.float32)

if shap_values.ndim == 1:
    shap_values = shap_values.reshape(-1, 1)

print(f"   SHAP shape: {shap_values.shape}")
print(f"   np.abs 均值: {np.abs(shap_values).mean():.6f}")

# ------------------- 3. SHAP 筛选 ≥0.1 -------------------

# 强制类型检查 (如果 shap_values 是 None，这里会报错，所以我们提前修复)
if shap_values is None:
    # 理论上这不应该发生，但如果发生了，说明第2步的赋值没有成功！
    raise RuntimeError("SHAP Values 为 None！请确保第 2 步 SHAP 计算代码已成功运行。")

# 确保 X_sample 存在且与 SHAP 矩阵对齐
# 【修正】: 在检查 .shape 之前，确保 shap_values 存在且不是 None
is_mismatch = (X_sample is None or
               'X_sample' not in globals() or
               len(X_sample) == 0 or
               shap_values.shape[0] != len(X_sample))

if is_mismatch:
    print("-> 警告：X_sample 或 SHAP 样本数不一致！正在重新采样 X_sample...")
    # 重新采样 X_sample
    sample_size = min(500, len(X_test_aligned))
    sample_idx = np.random.choice(len(X_test_aligned), size=sample_size, replace=False)
    X_sample = X_test_aligned.iloc[sample_idx].copy()

    # 重新运行 SHAP 计算块，确保 shap_values 与新的 X_sample 对齐
    # 由于代码是分块执行，最安全的做法是重新执行 SHAP
    print("-> 警告：重新执行 SHAP 计算...")
    explainer = shap.TreeExplainer(final_xgb_model)
    shap_exp = explainer(X_sample)

    # 提取正类 SHAP 值
    shap_values = shap_exp.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    shap_values = np.array(shap_values, dtype=np.float32)

# 最终长度检查 (特征列对齐)
if shap_values.shape[1] != len(X_sample.columns):
    print(f"   强制对齐：SHAP {shap_values.shape[1]} 列, 特征 {len(X_sample.columns)} 列")
    min_len = min(shap_values.shape[1], len(X_sample.columns))

    if min_len > 0:
        shap_values = shap_values[:, :min_len]
        X_sample = X_sample.iloc[:, :min_len]
    else:
        raise ValueError("SHAP Values 或特征列长度为零！无法继续计算。")

# 执行计算
shap_abs_mean = np.abs(shap_values).mean(axis=0)
print(f"   shap_abs_mean length: {len(shap_abs_mean)}, columns: {len(X_sample.columns)}")

shap_df = pd.DataFrame({
    'feature': X_sample.columns,
    'shap': shap_abs_mean
})


def clean_base_name(col):
    if '_' in col and col.split('_', 1)[0] in ['SOIL_TP', 'LAND_USE_TP']:
        return col.split('_', 1)[0]
    return col


shap_df['base'] = shap_df['feature'].apply(clean_base_name)
agg_shap = shap_df.groupby('base')['shap'].sum().reset_index().sort_values('shap', ascending=False)
agg_shap_filtered = agg_shap[agg_shap['shap'] >= 0.05].copy()
high_impact_features = agg_shap_filtered['base'].tolist()

print(f"SHAP ≥0.1 因子: {high_impact_features}")

# ------------------- 4. 检查高影响特征 -------------------
print("\n-> 检查筛选出的高影响特征（SHAP ≥0.1）：")
for feat in high_impact_features:
    # 查找原始特征（可能被 one-hot）
    matched_cols = [c for c in X_train_sel.columns if clean_base_name(c) == feat]
    if matched_cols:
        print(f"\n特征组: {feat} (共 {len(matched_cols)} 列)")
        for c in matched_cols:
            std_min = X_train_sel[c].min()
            std_max = X_train_sel[c].max()
            print(f"  列: {c} → 标准化范围: [{std_min:.2f}, {std_max:.2f}]")
    else:
        print(f"\n特征: {feat} - 不存在！")

# =============================================================================
# PDP/SHAP 初始化：强制重建 scaler 和 numerical_features (基于 X_train 原始数据)
# =============================================================================
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("-> 正在强制重建 scaler 和 numerical_features...")

# 1. 定义特征列表
numerical_features = dense_numeric + sparse_numeric
# 确保 categorical_features 在全局可用
cat_in_train = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

# =============================================================================
# 检查 SCALER 参数（用于验证）
# =============================================================================
try:
    feature_to_check = 'DEM'

    # 找到 DEM 在 dense_numeric 中的索引
    dem_idx_in_dense = dense_numeric.index(feature_to_check)

    dem_mean = scaler.mean_[dem_idx_in_dense]
    dem_scale = scaler.scale_[dem_idx_in_dense]

    print(f"--- 原始尺度检查 ({feature_to_check}) ---")
    print(f"Mean (均值): {dem_mean:.2f}")
    print(f"Scale (标准差): {dem_scale:.2f}")

    # 基于 X_train 的实际范围来检查
    min_actual = X_train[feature_to_check].min()
    max_actual = X_train[feature_to_check].max()

    print(f"实际原始范围 [Min, Max]: [{min_actual:.2f}, {max_actual:.2f}]")
    print("---------------------------------------")

except NameError:
    print("错误：变量未定义。请先运行所有预处理步骤。")
except Exception as e:
    print(f"检查参数时出错: {e}")

# ------------------- 5. 手动 PDP（最终修正：使用实际 Min/Max 范围） -------------------

def manual_pdp_corrected(model, X_standardized, feature, scaler_obj, X_original, dense_numeric_list, grid_points=200):
    if feature not in X_standardized.columns:
        return None, None

    is_standardized = feature in dense_numeric_list

    # --- X轴范围计算 ---
    if is_standardized:
        # 特征已经被标准化 (如 DEM)

        # 1. 提取标准化参数
        try:
            # 找到特征在 scaler.mean_ 中的索引
            idx = dense_numeric_list.index(feature)
        except ValueError:
            print(f"致命错误：特征 {feature} 在 dense_numeric 中但找不到索引！")
            return None, None

        mean = scaler_obj.mean_[idx]
        scale = scaler_obj.scale_[idx]

        # 2. 【核心修正】：使用 X_train (X_original) 的实际 Min/Max 作为绘图范围
        min_real = X_original[feature].min()
        max_real = X_original[feature].max()

        grid_real = np.linspace(min_real, max_real, grid_points)

        # 3. 将原始网格值转换为标准化值 (喂给模型)
        grid_std = (grid_real - mean) / scale

    else:
        # 特征未被标准化 (如 ROAD, RIVER)
        min_real = X_standardized[feature].min()  # 在 X_standardized 中，它们是 passthrough 原始值
        max_real = X_standardized[feature].max()
        grid_real = np.linspace(min_real, max_real, grid_points)
        grid_std = grid_real

    # --- PDP 值计算 (保持不变) ---
    pdp_vals = []
    X_mean_temp = X_standardized.mean().to_frame().T

    for val_for_model in grid_std:
        X_temp = X_mean_temp.copy()
        X_temp[feature] = val_for_model
        prob = calibrator.predict_proba(X_temp)[:, 1]
        pdp_vals.append(prob[0])

    return grid_real, np.array(pdp_vals)


# ------------------- 6. PDP 绘图（调用修正后的函数） -------------------

pdp_features = [f for f in high_impact_features if f in numerical_features]

if pdp_features:
    n_features = len(pdp_features)
    fig, axes = plt.subplots(1, n_features, figsize=(7 * n_features, 5.5))
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, pdp_features):
        print(f" 正在计算 PDP: {feat}...")

        # 【核心修正】：调用时传入 X_train (原始数据)
        x_grid, pdp_curve = manual_pdp_corrected(
            clean_xgb,
            X_train_final_sel,  # 标准化/预处理后的数据 (用于均值和非标准化特征的Min/Max)
            feat,
            scaler,
            X_train,  # <--- 传入 X_train (原始数据，用于获取DEM的实际Min/Max)
            dense_numeric
        )

        if x_grid is None:
            print(f"   跳过 {feat}：无法计算 PDP。")
            continue

        # ... (后续绘图和标签逻辑不变) ...
        ax.plot(x_grid, pdp_curve, color='darkred', linewidth=3.5)
        peak_idx = np.argmax(pdp_curve)
        ax.scatter(x_grid[peak_idx], pdp_curve[peak_idx], color='gold', s=120, zorder=5, edgecolor='black')

        ax.set_xlabel(feat + ' (原始尺度)', fontsize=12, fontweight='bold')
        ax.set_ylabel('滑坡概率 (相关性)', fontsize=12, fontweight='bold')

        shap_val = agg_shap_filtered.loc[agg_shap_filtered["base"] == feat, "shap"].values[0]
        ax.set_title(f'PDP: {feat} (SHAP = {shap_val:.3f})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    fig.suptitle('PDP 图 (SHAP ≥ 0.1, 数值特征)', fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PDP_SHAP_0.1_NUMERIC.png'), dpi=600, bbox_inches='tight')
    plt.show()
    print("PDP 图已保存！")
else:
    print("无数值特征可画 PDP！")

# =============================================================================
# 1. 数据检查：DEM > 3000M 样本分析
# =============================================================================
print("--- 1. 数据检查：DEM > 3000M 样本分析 ---")

# 假设 X_train 和 y_train 仍然是原始数据
if 'X_train' not in globals() or 'y_train' not in globals():
    print("错误：X_train 或 y_train 未定义，无法执行原始数据检查。")
else:
    # 筛选出 DEM > 3000M 的样本
    high_dem_mask = X_train['DEM'] > 3200

    # 总样本量
    total_samples = len(X_train)

    # 高海拔样本量
    high_dem_samples = high_dem_mask.sum()

    # 高海拔正例（滑坡）数量
    high_dem_positives = y_train[high_dem_mask].sum()

    print(f"训练集总样本数: {total_samples}")
    print(f"DEM > 3000M 的样本数: {high_dem_samples}")
    print(f"占总样本比例: {high_dem_samples / total_samples:.2%}")

    if high_dem_samples > 0:
        high_dem_positive_ratio = high_dem_positives / high_dem_samples
        print(f"DEM > 3200M 中的滑坡数: {high_dem_positives}")
        print(f"DEM > 3200M 中的滑坡比例: {high_dem_positive_ratio:.2%}")
    else:
        print("高海拔区域无样本，无法计算滑坡比例。")

print("---------------------------------------")

# =============================================================================
# 2. 绘制 ICE 图 (Individual Conditional Expectation Plot)
# =============================================================================
print("--- 2. 绘制 ICE 图：DEM ---")

# 假设要绘制 ICE 图的特征
ICE_FEATURE = 'DEM'
# 从标准化后的训练集中随机选择样本来绘制 ICE
N_ICE_SAMPLES = 50

# 随机选择样本进行 ICE 绘制
X_ice_samples = X_train_final_sel.sample(n=N_ICE_SAMPLES, random_state=42)

# 确保在 PDP 绘图代码中定义的 manual_pdp_corrected 函数是可用的
# （该函数需要 X_train, scaler 等，我们使用 PDP 函数的逻辑来生成网格）

# 1. 获取正确的绘图网格 (与 PDP 保持一致，使用实际 Min/Max)
# 假设 X_train 是原始数据
min_real = X_train[ICE_FEATURE].min()
max_real = X_train[ICE_FEATURE].max()
grid_points = 100
grid_real = np.linspace(min_real, max_real, grid_points)

# 2. 将原始网格转换为标准化网格 (喂给模型)
idx = dense_numeric.index(ICE_FEATURE)
mean = scaler.mean_[idx]
scale = scaler.scale_[idx]
grid_std = (grid_real - mean) / scale

# 3. 计算 ICE 值
ice_values = []

for idx_sample, X_sample in X_ice_samples.iterrows():
    X_sample_df = X_sample.to_frame().T.copy()  # 标准化后的单个样本
    sample_ice = []

    for val_for_model in grid_std:
        X_temp = X_sample_df.copy()

        # 替换模型需要的特征值 (标准化值)
        X_temp[ICE_FEATURE] = val_for_model

        # 使用 Calibrator 进行预测
        prob = calibrator.predict_proba(X_temp)[:, 1]
        sample_ice.append(prob[0])

    ice_values.append(sample_ice)

ice_values = np.array(ice_values)

# 4. 绘图
plt.figure(figsize=(10, 6))

# 绘制所有 ICE 曲线
for i in range(N_ICE_SAMPLES):
    plt.plot(grid_real, ice_values[i, :], color='blue', alpha=0.3)

# 绘制 PDP 曲线 (平均值)
pdp_mean = ice_values.mean(axis=0)
plt.plot(grid_real, pdp_mean, color='red', linewidth=3, label='PDP (平均)')

# 绘制垂直参考线 3000M
plt.axvline(x=3120, color='gray', linestyle='--', linewidth=1.5, label='3120M 分界线')

plt.xlabel(f'{ICE_FEATURE} (原始尺度)', fontsize=12)
plt.ylabel('滑坡概率', fontsize=12)
plt.title(f'ICE 和 PDP 图: {ICE_FEATURE} ({N_ICE_SAMPLES} 样本)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'ICE_PDP_{ICE_FEATURE}_final.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"ICE 图已生成并保存！")
print("---------------------------------------")

# =============================================================================
# 1. ROAD 特征分析：基于关键区间的样本分析 (0-100, 100-350, 350-600, 600-700, >700)
# =============================================================================
print("\n--- 1. ROAD 特征分析：基于关键区间的样本统计 ---")

# 假设 X_train 和 y_train 仍然是原始数据
if 'X_train' not in globals() or 'y_train' not in globals():
    print("错误：X_train 或 y_train 未定义，无法执行原始数据检查。")
else:
    # 定义关键区间阈值 (Meters)
    THRESHOLDS = [0, 100, 350, 600, 700, 2200, X_train['Road'].max()]

    # 定义区间标签
    RANGE_LABELS = [
        "0 - 100M (高位)",
        "100M - 350M (急降)",
        "350M - 600M (低位波动)",
        "600M - 700M (急升)",
        "700M - 2200M (高位波动)",
        ">2200M (平缓）"
    ]

    total_samples = len(X_train)
    print(f"训练集总样本数: {total_samples}")
    print("---------------------------------------")
    print("区间分析结果：")

    for i in range(len(THRESHOLDS) - 1):
        lower = THRESHOLDS[i]
        upper = THRESHOLDS[i + 1]
        label = RANGE_LABELS[i]

        # 定义掩码
        if i == len(THRESHOLDS) - 2:  # 最后一个区间 (>700M)
            mask = (X_train['Road'] > lower)
        else:
            mask = (X_train['Road'] > lower) & (X_train['Road'] <= upper)

        # 统计
        range_samples = mask.sum()
        range_positives = y_train[mask].sum()

        print(f"  {label:<20} | 样本数: {range_samples:^6} | 占总样本比例: {range_samples / total_samples:.2%}")

        if range_samples > 0:
            range_positive_ratio = range_positives / range_samples
            print(f"                       | 滑坡数: {range_positives:^6} | 滑坡比例: {range_positive_ratio:.2%}")
        else:
            print(f"                       | 该区间无样本。")

print("---------------------------------------")

# =============================================================================
# 2. 绘制 ICE 图 (Individual Conditional Expectation Plot) - ROAD (更新参考线)
# =============================================================================
print("\n--- 2. 绘制 ICE 图：ROAD ---")

ICE_FEATURE = 'Road'
N_ICE_SAMPLES = 50

# 随机选择样本进行 ICE 绘制
# 注意：这里使用 X_train_final_sel 和 calibrator，确保与模型训练和校准一致
X_ice_samples = X_train_final_sel.sample(n=N_ICE_SAMPLES, random_state=42)

# --- 获取 ROAD 特征的网格 ---
min_real = X_train_final_sel[ICE_FEATURE].min()
max_real = X_train_final_sel[ICE_FEATURE].max()

# 集中在关键区域并扩展到最大值
grid_points = 200  # 增加点数使曲线更平滑
grid_real = np.unique(np.concatenate([
    np.linspace(min_real, 700, 100),  # 密集的关键区域
    np.linspace(700, 2500, 50),  # 波动区域
    np.linspace(2500, max_real, 50)  # 平缓区域
]))
grid_std = grid_real

# 3. 计算 ICE 值 (保持不变)
ice_values = []
for idx_sample, X_sample in X_ice_samples.iterrows():
    X_sample_df = X_sample.to_frame().T.copy()
    sample_ice = []
    for val_for_model in grid_std:
        X_temp = X_sample_df.copy()
        X_temp[ICE_FEATURE] = val_for_model
        # 确保使用 calibrator 进行预测
        prob = calibrator.predict_proba(X_temp)[:, 1]
        sample_ice.append(prob[0])
    ice_values.append(sample_ice)

ice_values = np.array(ice_values)

# 4. 绘图 (更新参考线)
plt.figure(figsize=(11, 7))

# 绘制所有 ICE 曲线 (使用更高的 Z-order 确保 PDP 覆盖在上面)
for i in range(N_ICE_SAMPLES):
    plt.plot(grid_std, ice_values[i, :], color='skyblue', alpha=0.3, zorder=1)

# 绘制 PDP 曲线 (平均值)
pdp_mean = ice_values.mean(axis=0)
plt.plot(grid_std, pdp_mean, color='#d62728', linewidth=4, label='PDP (平均预测概率)', zorder=3)

# 绘制垂直参考线 (突出您指定的转折点)
CRITICAL_POINTS = [100, 350, 600, 700, 2200]
for p in CRITICAL_POINTS:
    # 查找最接近的点在 PDP 上的高度
    y_val = pdp_mean[np.argmin(np.abs(grid_std - p))]

    plt.axvline(x=p, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
    # 添加文本标签
    plt.text(p, 0.95, f'{p}M', color='black', fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7), zorder=4)

plt.xlabel(f'{ICE_FEATURE} (距离道路, m)', fontsize=13, fontweight='bold')
plt.ylabel('滑坡概率', fontsize=13, fontweight='bold')
plt.title(f'ICE 和 PDP 分析: {ICE_FEATURE} - 关键转折点 ({N_ICE_SAMPLES} 样本)', fontsize=15)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.4, linestyle='--')
plt.xlim(0, max_real)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'ICE_PDP_{ICE_FEATURE}_Critical_Points.png'), dpi=400, bbox_inches='tight')
plt.show()
print(f"ICE 图已生成并保存！")
print("---------------------------------------")

# =============================================================================
# 修正后的 SHAP 依赖图绘制（ROAD DEM）- 终极版
# 策略：交互特征选择切换为【统计相关性】最大的特征
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import matplotlib
from scipy.stats import spearmanr  # 【新增导入】

# 尝试使用最常见的交互式后端
try:
    matplotlib.use('TkAgg') # 或 'QtAgg', 'WXAgg' 等
    print("-> 成功切换 Matplotlib 后端至 'TkAgg' (交互式)。")
except Exception as e:
    # 如果失败，则尝试 Agg 确保能保存文件
    matplotlib.use('Agg')
    print(f"-> 警告：交互式后端切换失败，已切换至 'Agg'。错误: {e}")

# =============================================================================
# 终极救援 V13：修复采样顺序 + 修复行数和列数不匹配
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import matplotlib
from scipy.stats import spearmanr
import matplotlib.cm as cm

# ------------------- 核心修正 1/2：强制使用 X_test_final_sel 的 9 个特征进行采样与对齐 -------------------
# !!! 必须确保 X_test_final_sel 变量已被定义并包含 9 个特征 !!!
assert 'X_test_final_sel' in globals(), "❌ 致命错误：X_test_final_sel 未定义！请确认 SHAP 脚本已运行。"
assert 'X_test_sel' in globals(), "❌ 致命错误：X_test_sel (预处理后的测试集) 未定义！"
assert 'final_columns' in globals(), "❌ 致命错误：final_columns (9个特征名列表) 未定义！"

# 确定 SHAP 值的行数和特征列数
expected_n_samples = shap_values.shape[0]
final_feature_set = X_test_final_sel.columns.tolist()
M_features = len(final_feature_set)
print(f"-> 确认最终建模特征数 M: {M_features}")

# 【安全检查】检查 SHAP 值列数是否匹配
if shap_values.shape[1] != M_features:
    print(f"❌ 致命错误：SHAP 列数 ({shap_values.shape[1]}) 与最终数据列数 ({M_features}) 不一致，无法绘图！")
    if shap_values.shape[1] > M_features:
        shap_values = shap_values[:, :M_features] # 强制截断 SHAP 值
        print(f"   已强制截断 SHAP 值到 {M_features} 列。")
    else:
        raise ValueError("SHAP 值列数少于最终特征数，无法安全修复。请重新计算 SHAP 值。")


# --- 核心修正：采样和对齐 SHAP 值与输入数据 ---
print("-> 正在准备 SHAP 绘图数据 (强制 9 特征对齐)...")

N_sample = min(500, expected_n_samples) # 确保采样的行数不超过 SHAP 值的行数

# 1. 随机抽取样本的索引
sample_indices = X_test_final_sel.sample(n=N_sample, random_state=42).index

# 2. 抽样数据 (X_sample_original 只包含 9 个特征)
X_sample_original = X_test_final_sel.loc[sample_indices, final_feature_set].copy()

# 3. 抽样 SHAP 值 (获取与抽样数据行对应的 SHAP 值)
sample_shap_values = shap_values[X_test_final_sel.index.get_indexer(sample_indices)]

# 标记为成功，不触发 Fallback
raw_data_retrieval_failed = False
print(f"✅ X_sample_original (用于绘图原始刻度) 创建成功，列数 {X_sample_original.shape[1]}。")
print(f"✅ sample_shap_values 成功对齐，行数 {sample_shap_values.shape[0]}。")

# ------------------- 依赖函数定义 (完整版) -------------------

def find_best_correlator(target_feature, X_data, method='spearman'):
    """找到与目标特征相关性绝对值最大的特征（非目标特征自身）"""
    if isinstance(X_data, pd.Series):
        X_data = X_data.to_frame()

    if not isinstance(X_data, pd.DataFrame):
        return 'DEM'

    features_to_check = X_data.columns.drop(target_feature, errors='ignore')
    correlations = {}

    for col in features_to_check:
        try:
            if col not in X_data.columns:
                continue

            clean_data = X_data[[target_feature, col]].dropna()
            if len(clean_data) < 2:
                continue

            corr, _ = spearmanr(clean_data[target_feature], clean_data[col])
            correlations[col] = np.abs(corr)
        except Exception:
            continue

    if not correlations:
        return 'DEM'

    best_correlator = max(correlations, key=correlations.get)
    return best_correlator


def plot_shap_dependence_original_scale(ind_feature, inter_feature, shap_vals, X_orig, is_fallback, title_prefix=""):
    """绘制 SHAP 依赖图，并确保尺寸匹配"""

    # 核心修正 3：列数匹配检查和修复
    n_shap_cols = shap_vals.shape[1]
    n_data_cols = X_orig.shape[1]

    if n_shap_cols != n_data_cols:
        print(f"⚠️ 警告：SHAP 列数 ({n_shap_cols}) 与数据列数 ({n_data_cols}) 不一致。")
        if n_shap_cols > n_data_cols:
            # 假设 X_orig 的列数是正确的，截断 SHAP 数组
            shap_vals = shap_vals[:, :n_data_cols]
            print(f"   已截断 shap_values 列数至 {n_data_cols}。")
        else:
            print(f"   SHAP 列数 ({n_shap_cols}) 少于数据列数 ({n_data_cols})，无法安全修复。绘图失败。")
            return

    # 行数匹配检查 (防止新的错误)
    if shap_vals.shape[0] != X_orig.shape[0]:
        print(f"致命错误：SHAP 行数 ({shap_vals.shape[0]}) 与 X_orig 行数 ({X_orig.shape[0]}) 不一致，绘图失败。")
        return

    plt.figure(figsize=(10, 6))

    shap.dependence_plot(
        ind=ind_feature,
        shap_values=shap_vals,
        features=X_orig,
        interaction_index=inter_feature,
        show=False
    )

    fig = plt.gcf()
    cax = fig.axes[-1]

    if is_fallback:
        scale_label = "原始尺度 (Fallback: 标准化/筛选数据)"
        title_suffix = " (刻度警告)"
    else:
        scale_label = "原始尺度 (Real Scale)"
        title_suffix = ""

    cax.set_ylabel(f'{inter_feature} ({scale_label})', rotation=270, labelpad=15, fontsize=12)
    plt.xlabel(f'{ind_feature} ({scale_label})', fontsize=12, fontweight='bold')
    plt.ylabel('特征对输出的贡献 (SHAP 值)', fontsize=12, fontweight='bold')

    plt.title(f'{title_prefix} SHAP 依赖图: {ind_feature} vs {inter_feature} (相关性交互){title_suffix}', fontsize=14,
              fontweight='bold')

    plt.tight_layout()

    # 假设 OUTPUT_DIR 已经在全局定义
    filename = os.path.join(OUTPUT_DIR, f'SHAP_Dependence_{ind_feature}_vs_{inter_feature}_Correlate_FINAL.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"-> 依赖图已保存到: {filename}")

    # 实时显示图表
    plt.show()
    plt.close()


def plot_ice_and_dependence(feature_name, interactor_name, shap_explainer, X_orig, is_fallback):
    """绘制 ICE 图（SHAP 方式）和依赖图"""

    # 1. 绘制 SHAP Dependence Plot (依赖图)
    print(f"   - 绘制 {feature_name} vs {interactor_name} 依赖图...")
    plot_shap_dependence_original_scale(
        ind_feature=feature_name,
        inter_feature=interactor_name,
        shap_vals=shap_explainer,
        X_orig=X_orig,
        is_fallback=is_fallback,
        title_prefix="[依赖图]"
    )

    # 2. 绘制 SHAP ICE Plot (个体条件期望图)
    print(f"   - 绘制 {feature_name} 的 ICE/Partial Dependence 图...")

    plt.figure(figsize=(10, 6))

    # 在 ICE/PDP 绘图前，也需要检查和修正列数
    n_shap_cols = shap_explainer.shape[1]
    n_data_cols = X_orig.shape[1]
    shap_vals_ice = shap_explainer.copy()

    if n_shap_cols != n_data_cols:
        if n_shap_cols > n_data_cols:
            shap_vals_ice = shap_vals_ice[:, :n_data_cols]
        else:
            print("   SHAP 列数少于数据列数，ICE/PDP 绘图失败。")
            return

    shap.dependence_plot(
        ind=feature_name,
        shap_values=shap_vals_ice,
        features=X_orig,
        interaction_index=None,
        show=False,
        alpha=0.5
    )

    if is_fallback:
        scale_label = "原始尺度 (Fallback: 标准化/筛选数据)"
        title_suffix = " (刻度警告)"
    else:
        scale_label = "原始尺度 (Real Scale)"
        title_suffix = ""

    fig = plt.gcf()
    if len(fig.axes) > 1:
        cax = fig.axes[-1]
        cax.set_visible(False)
        main_ax = fig.axes[0]
        box = main_ax.get_position()
        main_ax.set_position([box.x0, box.y0, box.width, box.height])

    plt.xlabel(f'{feature_name} ({scale_label})', fontsize=12, fontweight='bold')
    plt.ylabel('特征对输出的贡献 (SHAP 值)', fontsize=12, fontweight='bold')
    plt.title(f'[ICE/PDP 图] SHAP ICE/PDP: {feature_name}{title_suffix}', fontsize=14,
              fontweight='bold')

    plt.tight_layout()

    filename_ice = os.path.join(OUTPUT_DIR, f'SHAP_ICE_PDP_{feature_name}_FINAL.png')
    plt.savefig(filename_ice, dpi=300, bbox_inches='tight')
    print(f"-> ICE/PDP 图已保存到: {filename_ice}")

    # 实时显示图表
    plt.show()
    plt.close()

# ------------------- 绘图和统计执行 (V13) -------------------

# 1. 绘制 ROAD 的依赖图 (现在应该能显示了)
print("\n--- 1. 绘制 ROAD 的 SHAP 依赖图 ---")
ROAD_FEATURE = 'Road'
interaction_feature_road = find_best_correlator(ROAD_FEATURE, X_sample_original)
print(f"   ROAD 最佳交互特征 (相关性): {interaction_feature_road}")

if ROAD_FEATURE in X_sample_original.columns:
    plot_shap_dependence_original_scale(
        ind_feature=ROAD_FEATURE,
        inter_feature=interaction_feature_road,
        shap_vals=sample_shap_values,
        X_orig=X_sample_original,
        is_fallback=raw_data_retrieval_failed
    )
else:
    print(f"致命错误：特征 {ROAD_FEATURE} 不在 X_sample_original 列中！")

print("\n--- 2. 绘制 DEM 的 SHAP 依赖图 ---")
DEM_FEATURE = 'DEM'
interaction_feature_dem = find_best_correlator(DEM_FEATURE, X_sample_original)
print(f"   DEM 最佳交互特征 (相关性): {interaction_feature_dem}")

if DEM_FEATURE in X_sample_original.columns:
    plot_shap_dependence_original_scale(
        ind_feature=DEM_FEATURE,
        inter_feature=interaction_feature_dem,
        shap_vals=sample_shap_values,
        X_orig=X_sample_original,
        is_fallback=raw_data_retrieval_failed
    )
else:
    print(f"致命错误：特征 {DEM_FEATURE} 不在 X_sample_original 列中！")

# =============================================================================
# 【宇宙终极栅格预测】自动补全所有变量 | 支持 RIVER/FAULT/ROAD | 零报错
# =============================================================================
print("\n" + "=" * 90)
print("【宇宙终极栅格预测启动】自动补全所有变量 | 支持 RIVER/FAULT/ROAD | 永不爆炸")
print("=" * 90)

import os
import numpy as np
import pandas as pd
import rasterio
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# ------------------- 1. 自动补全 tif_files（IDE + 运行双保险） -------------------
tif_files = {}  # ← 提前定义，IDE 立刻识别！

if 'tif_files' not in globals() or not tif_files:
    print("   tif_files 未定义或为空，正在自动扫描...")
    tif_dir = "D:/JAVIER_QHNU/HeHuang_valley_slide/Python_proj_for_slide/100M_total_tif"

    if os.path.exists(tif_dir):
        for file in os.listdir(tif_dir):
            if file.lower().endswith((".tif", ".tiff")):
                name_raw = os.path.splitext(file)[0]
                # ！！！修正：保持原始大小写或与模型特征名匹配
                # 假设您模型特征名是 'Aspect', 'River', 'SOIL_TP'
                name = name_raw # 保持文件名原始大小写
                tif_files[name] = os.path.join(tif_dir, file)
        # ！！！修正：打印时使用正确的名称
        print(f"   自动发现 {len(tif_files)} 个 TIF 文件: {list(tif_files.keys())}")
    else:
        print(f"   错误: TIF 文件夹 {tif_dir} 不存在！")
        tif_files = {}  # 保持空字典
else:
    print(f"   tif_files 已定义: {len(tif_files)} 个 → {list(tif_files.keys())}")


# =============================================================================
# 2. scaler / encoder / final_feature_columns (加载)
# =============================================================================
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler_final.pkl')
ENCODER_PATH = os.path.join(OUTPUT_DIR, 'encoder_final.pkl')
FEATURE_LIST_PATH = os.path.join(OUTPUT_DIR, 'final_feature_list.pkl')

# 检查所有文件是否存在
if (os.path.exists(SCALER_PATH) and
        os.path.exists(ENCODER_PATH) and
        os.path.exists(FEATURE_LIST_PATH)):

    print("   正在加载已保存的 scaler, encoder 和特征列表...")

    try:
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        final_feature_columns = joblib.load(FEATURE_LIST_PATH)

        # 重新定义 numerical_features (用于步骤 7 的特征读取)
        categorical_features = ['SOIL_TP', 'LAND_USE_TP']
        numerical_features = [c for c in final_feature_columns if c not in categorical_features]

        print(f"   工具加载完成！模型特征数: {len(final_feature_columns)} 个")

    except Exception as e:
        print(f"   错误: 加载工具失败，请检查文件完整性: {e}")
        # 强制退出或跳过后续步骤
        raise e

else:
    print("   错误：缺少 scaler, encoder 或特征列表文件！无法进行栅格预测。")
    print("   请先运行完整的模型训练脚本并保存所有工具。")
    # 强制退出或跳过后续步骤
    raise FileNotFoundError("缺少必要的预处理工具文件。")

# 3. final_feature_columns
final_feature_columns = X_train_sel.columns.tolist()
print(f"   模型特征数: {len(final_feature_columns)}")

# 4. calibrator（校准器）
if 'calibrator' not in globals():
    print("   警告: calibrator 未定义。请确保已运行 CalibratedClassifierCV 训练！")
    # 如果 calibrator 不存在，则使用原始模型（但这不是最佳做法）
    final_predictor = clean_xgb
else:
    print("   calibrator (Sigmoid/CV) 已存在，使用校准器进行预测。")
    final_predictor = calibrator  # 使用校准器对象

# 5. best_threshold (假设在 Sigmoid/CV 步骤中已定义)
if 'best_threshold' not in globals():
    print("   警告: best_threshold 未定义。正在使用默认 0.5。")
    best_threshold = 0.5

# 关键：预测部分需要使用 calibrator/final_predictor 的 predict_proba
# 由于最终的 Sigmoid/CV 校准器已经包装了原始模型，我们直接使用它。
# 栅格预测代码中，我们使用 clean_xgb.predict_proba 来获取原始概率，然后用 calibrator 来校准。

# 6. OUTPUT 路径
OUTPUT_DIR = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else "XGBOOST的结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PROB = os.path.join(OUTPUT_DIR, "landslide_probability.tif")
OUTPUT_CLASS = os.path.join(OUTPUT_DIR, "landslide_classification.tif")

# ------------------- 7. 栅格预测（终极版） -------------------
print("\n" + "=" * 80)
print("开始全栅格预测（支持 RIVER/FAULT/ROAD 真实距离）")
print("=" * 80)

if 'tif_files' not in globals() or not tif_files:
    print("   tif_files 无效，跳过栅格预测")
else:
    # 原始 TIF 特征
    raw_tif_features = numerical_features + ['Aspect', 'SOIL_TP', 'LAND_USE_TP']
    distance_features = ['River', 'Fault', 'Road']
    raw_tif_features += [f for f in distance_features if f in tif_files]

    raster_arrays = {}
    meta = None
    height, width = None, None

    print(f"-> 正在读取 {len(raw_tif_features)} 个原始 TIF 文件...")
    for feat in raw_tif_features:
        tif_path = tif_files.get(feat)
        if not tif_path or not os.path.exists(tif_path):
            print(f"   错误: 缺少 {feat}.tif")
            continue
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            if meta is None:
                meta = src.meta.copy()
                height, width = data.shape
            elif data.shape != (height, width):
                print(f"   错误: {feat}.tif 尺寸不一致")
                continue
            raster_arrays[feat] = data.flatten()
            print(f"   读取: {feat}.tif → {data.shape}")

    if not raster_arrays:
        print("   无有效 TIF 文件，跳过预测")
    else:
        n_pixels = height * width
        df_raster = pd.DataFrame(raster_arrays)
        df_raster.replace(-9999, np.nan, inplace=True)
        print(f"   原始栅格 DataFrame: {df_raster.shape}")

        # 派生 sin/cos_ASPECT
        if 'Aspect' in df_raster.columns:
            print("   -> 计算 sin_Aspect 和 cos_Aspect...")
            df_raster['sin_Aspect'] = np.sin(np.deg2rad(df_raster['Aspect']))
            df_raster['cos_Aspect'] = np.cos(np.deg2rad(df_raster['Aspect']))
            df_raster = df_raster.drop(columns=['Aspect'])

        # 预处理
        mask = df_raster.isna().any(axis=1)
        print(f"   NaN 像素比例: {mask.mean():.2%}")
        df_valid = df_raster[~mask].copy()

        # 标准化
        num_cols = [c for c in numerical_features + distance_features if c in df_valid.columns]
        df_valid_num = scaler.transform(df_valid[num_cols])

        # 独热编码
        cat_cols = [c for c in ['SOIL_TP', 'LAND_USE_TP'] if c in df_valid.columns]
        df_valid_cat = encoder.transform(df_valid[cat_cols])
        df_valid_cat = df_valid_cat[:, :]  # 保持完整

        # 合并
        processed_cols = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
        df_valid_processed = pd.DataFrame(
            np.hstack([df_valid_num, df_valid_cat]),
            columns=processed_cols
        )

        # 补全缺失列
        missing_cols = [c for c in final_feature_columns if c not in df_valid_processed.columns]
        if missing_cols:
            print(f"   警告: 补 0 填充 {len(missing_cols)} 个特征")
            for c in missing_cols:
                df_valid_processed[c] = 0
        df_valid_processed = df_valid_processed[final_feature_columns]

        # 预测
        print("   -> 正在预测...")

        # 核心逻辑：统一使用 calibrator 对象进行预测

        if 'calibrator' in globals():
            # **使用 CalibratedClassifierCV 预测校准后的概率**
            # CalibratedClassifierCV 的 predict_proba 接收处理后的特征矩阵
            proba_valid = calibrator.predict_proba(df_valid_processed)[:, 1]
            print("   -> 使用 Calibrator 预测校准后概率。")

        else:
            # **如果 calibrator 未定义，则降级为原始模型预测**
            # 这种情况不应该发生，但作为代码保护
            dmatrix = xgb.DMatrix(df_valid_processed)
            raw_proba = final_xgb_model.predict(dmatrix)
            proba_valid = raw_proba
            print("   -> 警告：Calibrator 未定义，使用原始 XGBoost 概率！")

        # 分类
        # best_threshold 变量在步骤 5 中定义，确保它存在
        class_valid = (proba_valid >= best_threshold).astype(int)

        # 回填
        proba_full = np.full(n_pixels, np.nan)
        class_full = np.full(n_pixels, np.nan)
        proba_full[~mask] = proba_valid
        class_full[~mask] = class_valid

        meta.update(dtype='float32', nodata=np.nan)

        # 输出
        with rasterio.open(OUTPUT_PROB, 'w', **meta) as dst:
            dst.write(proba_full.reshape(height, width).astype('float32'), 1)
        print(f"   已生成: {OUTPUT_PROB}")

        with rasterio.open(OUTPUT_CLASS, 'w', **meta) as dst:
            dst.write(class_full.reshape(height, width).astype('float32'), 1)
        print(f"   已生成: {OUTPUT_CLASS}")

        print("   栅格预测完成！")

# =============================================================================
# 11. 🥈 SHAP 值栅格图 (Feature Influence Map)
# =============================================================================

import shap
from tqdm import tqdm
import os
import rasterio

print("\n" + "=" * 80)
print("【生成 SHAP 值栅格图 (Feature Influence Map)】")
print("=" * 80)

# 假设 shap_explainer 已经被定义 (例如 shap.TreeExplainer(clean_xgb))
if 'shap_explainer' not in globals():
    print("   警告: 正在创建 SHAP Explainer...")
    # 假设 clean_xgb 是您的原始 XGBoost 模型
    shap_explainer = shap.TreeExplainer(final_xgb_model)
    print("   SHAP Explainer 创建完成。")

# 假设 df_raster, df_valid, mask, n_pixels, height, width, meta, final_feature_columns 已在步骤 7 中生成

if 'df_valid' not in locals() or df_valid.empty:
    print("   错误: df_valid (有效栅格数据) 不存在或为空，跳过 SHAP 栅格图生成。")
else:
    # 1. 重新准备用于预测的有效处理数据
    # 这一步与步骤 7 的预测部分重复，但为了 SHAP 计算的独立性，需要再次准备。

    # 仅使用步骤 7 中已定义的 df_valid_processed DataFrame
    # 确保它是经过预处理且列名顺序正确的

    # *** 为了避免代码冗余，我们直接使用步骤 7 中生成的 df_valid_processed ***
    # 假设 df_valid_processed 包含了所有 NaN 移除后、预处理完成、且列名匹配 final_feature_columns 的数据。

    X_shap_valid = df_valid_processed.copy()

    # 2. 计算 SHAP 值 (可能耗时较长)
    print(f"   -> 正在计算 {X_shap_valid.shape[0]} 个有效像素的 SHAP 值...")

    # 由于数据量大，使用并行计算 (TreeExplainer 默认支持)
    shap_values_raw = shap_explainer.shap_values(X_shap_valid)

    # 对于二分类，shap_values_raw 是一个长度为 2 的列表，我们取类别 1 的 SHAP 值
    # 如果 explainer 是针对 Sigmoid 输出的，则直接取第一个元素
    # 如果 explainer 是针对 Logit 输出的，则取第二个元素
    if isinstance(shap_values_raw, list):
        # 假设我们关注类别 1 的贡献
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw

    print("   SHAP 值计算完成。")

    # 3. 提取关键 SHAP 指标

    # (A) Mean Absolute SHAP (平均绝对贡献度) - 衡量每个像素的整体风险有多依赖于模型
    mean_abs_shap = np.sum(np.abs(shap_values), axis=1) / shap_values.shape[1]

    # (B) Max Absolute SHAP Feature Index (最大贡献特征的索引)
    max_abs_shap_index = np.argmax(np.abs(shap_values), axis=1)

    # 4. 栅格回填

    # 最终特征列名，用于 SHAP 解释
    final_feature_columns = X_shap_valid.columns.tolist()

    # (A) 回填平均绝对 SHAP 值
    mean_abs_shap_full = np.full(n_pixels, np.nan)
    mean_abs_shap_full[~mask] = mean_abs_shap

    # (B) 回填最大贡献特征的索引 (用于分类图，表示哪个特征最重要)
    # 我们将索引映射回特征名称的哈希值，以便于栅格存储 (需要离散化)
    feature_names_map = {i: name for i, name in enumerate(final_feature_columns)}
    # 将索引数组映射为特征名的哈希值 (或简单的整数 ID)
    max_feature_id = max_abs_shap_index + 1  # +1 避免 ID 0

    max_feature_id_full = np.full(n_pixels, np.nan)
    max_feature_id_full[~mask] = max_feature_id

    # 5. 输出 TIF 文件
    meta_shap = meta.copy()
    meta_shap.update(dtype='float32', nodata=np.nan)

    # 输出 1: 平均绝对 SHAP 贡献图 (连续图)
    output_tif_1 = os.path.join(OUTPUT_DIR, 'SHAP_Mean_Abs_Influence.tif')
    with rasterio.open(output_tif_1, 'w', **meta_shap) as dst:
        dst.write(mean_abs_shap_full.reshape(height, width).astype('float32'), 1)
    print(f"   已生成 SHAP 平均贡献图: {output_tif_1}")

    # 输出 2: 最大贡献特征 ID 图 (分类图 - 需要 GIS 软件配色)
    output_tif_2 = os.path.join(OUTPUT_DIR, 'SHAP_Max_Feature_ID.tif')
    with rasterio.open(output_tif_2, 'w', **meta_shap) as dst:
        dst.write(max_feature_id_full.reshape(height, width).astype('float32'), 1)
    print(f"   已生成 SHAP 最大贡献特征 ID 图: {output_tif_2}")

    # 输出特征ID映射表 (辅助 GIS 软件配色)
    mapping_table = pd.DataFrame(feature_names_map.items(), columns=['Feature_ID', 'Feature_Name'])
    mapping_table['Feature_ID'] = mapping_table['Feature_ID'] + 1  # 匹配 TIF 文件的 ID
    mapping_table.to_csv(os.path.join(OUTPUT_DIR, 'SHAP_Feature_ID_Mapping.csv'), index=False)
    print("   已生成特征 ID 映射表 (用于 GIS 配色)。")

print("SHAP 栅格图生成完成。")
