import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\3_samples\point1__Merge_TableToExcel.xlsx"
OUTPUT_PATH = r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\3_samples\point1__samples_ml_ready.csv"

# ====== 读入 ======
p = Path(INPUT_PATH)
if p.suffix.lower() == ".csv":
    df = pd.read_csv(p)
elif p.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(p)
else:
    raise ValueError("只支持 .csv/.xlsx/.xls")

# ★ 很关键：防止导出后列名带空格
df.columns = [c.strip() for c in df.columns]

# ====== 你的实际列名（按截图） ======
COL_LABEL   = "label"
COL_ASPECT  = "Aspect"
COL_FLOWACC = "FlowAcc"
COL_SPI     = "SPI"
COL_FLOWDIR = "FlowDir"

# 建议：第一版模型先丢掉 D8 FlowDir（常引噪）
DROP_FLOWDIR = True

# SPI 常有负值：这里我默认做 log，并自动平移到可取 log 的范围
LOG_SPI = True

# （可选）你不想让 OBJECTID 进模型，就删掉
DROP_ID = True


# ====== 基础清洗 ======
df = df.copy()

# label 转数值并保留 0/1
df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce")
df = df[df[COL_LABEL].isin([0, 1])]

# 其他列尽量转成数值
for c in df.columns:
    if c != COL_LABEL:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 去掉含缺失值的行（ArcGIS 提取后 Null 会变 NaN）
df = df.dropna(axis=0).reset_index(drop=True)

# （可选）删 OBJECTID
if DROP_ID and "OBJECTID" in df.columns:
    df = df.drop(columns=["OBJECTID"])

# ====== 3.6-A：Aspect -> sin/cos ======
if COL_ASPECT in df.columns:
    asp = df[COL_ASPECT].to_numpy(dtype=float)
    asp = np.where(asp < 0, 0.0, asp)     # -1 平坦区处理成 0°
    asp_rad = np.deg2rad(asp)
    df["sin_aspect"] = np.sin(asp_rad)
    df["cos_aspect"] = np.cos(asp_rad)
    df = df.drop(columns=[COL_ASPECT])

# ====== 3.6-B：FlowAcc -> log1p ======
if COL_FLOWACC in df.columns:
    fa = df[COL_FLOWACC].to_numpy(dtype=float)
    fa = np.where(fa < 0, 0.0, fa)
    df["flowacc_log"] = np.log1p(fa)
    df = df.drop(columns=[COL_FLOWACC])

# ====== 3.6-B(可选)：SPI -> log（自动平移处理负值） ======
if LOG_SPI and (COL_SPI in df.columns):
    spi = df[COL_SPI].to_numpy(dtype=float)
    min_spi = np.nanmin(spi)
    shift = 0.0
    if min_spi <= -1.0:
        shift = -min_spi + 1.0
    elif min_spi < 0.0:
        shift = -min_spi + 1e-6
    df["spi_log"] = np.log(spi + shift + 1.0)
    df = df.drop(columns=[COL_SPI])

# ====== 3.6-C：丢 FlowDir（D8 编码） ======
if DROP_FLOWDIR and (COL_FLOWDIR in df.columns):
    df = df.drop(columns=[COL_FLOWDIR])

# ====== 输出：把 label 放最后 ======
cols = [c for c in df.columns if c != COL_LABEL] + [COL_LABEL]
df = df[cols]
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("✅ 输出：", OUTPUT_PATH)
print("样本数：", len(df), "特征数：", df.shape[1] - 1)
print("列名：", df.columns.tolist())
