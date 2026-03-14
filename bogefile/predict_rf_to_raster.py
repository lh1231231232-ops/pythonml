import numpy as np
import rasterio
import joblib

# ========= 0) 路径：改成你的 =========
MODEL_PATH = r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\4_weights\rf_model.joblib"
OUT_PATH   = r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\4_weights\susceptibility_prob.tif"

# 你的因子栅格路径（名字按你自己的文件改）
rasters = {
    "Slope":    r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\Slope.tif",
    # "Curvatu":  r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\Curvatu.tif",
    # "Roughnes": r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\Roughness.tif",
    # "TWI":      r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\TWI.tif",
    # "SPI":      r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\SPI.tif",
    # "FlowAcc":  r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\FlowAcc.tif",
    # "Aspect":   r"D:\Users\LHX\Desktop\论文\贡日嘎布\数据\2_factors\Aspect.tif",
}

# ========= 1) 加载模型 =========
rf = joblib.load(MODEL_PATH)

# ========= 2) 打开栅格并检查对齐 =========
srcs = {k: rasterio.open(v) for k, v in rasters.items()}
ref = next(iter(srcs.values()))

for k, s in srcs.items():
    if (s.width != ref.width) or (s.height != ref.height):
        raise ValueError(f"[对齐错误] {k} 尺寸不一致")
    if (s.transform != ref.transform) or (s.crs != ref.crs):
        raise ValueError(f"[对齐错误] {k} transform/crs 不一致（需重投影/重采样/对齐）")

profile = ref.profile.copy()
profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="lzw"
)

# ========= 3) 为 SPI 的 log 变换计算全局 shift（处理负值） =========
spi_src = srcs["SPI"]
spi_min = np.inf
for _, window in spi_src.block_windows(1):
    a = spi_src.read(1, window=window, masked=True)
    if a.count() > 0:
        spi_min = min(float(a.min()), spi_min)

if not np.isfinite(spi_min):
    spi_min = 0.0

shift = 0.0
if spi_min <= -1.0:
    shift = -spi_min + 1.0
elif spi_min < 0.0:
    shift = -spi_min + 1e-6

print(f"SPI min = {spi_min}, shift = {shift}")

# ========= 4) 分块预测并写出 =========
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    for _, window in ref.block_windows(1):
        slope   = srcs["Slope"].read(1, window=window, masked=True).astype("float32")
        curv    = srcs["Curvatu"].read(1, window=window, masked=True).astype("float32")
        rough   = srcs["Roughnes"].read(1, window=window, masked=True).astype("float32")
        twi     = srcs["TWI"].read(1, window=window, masked=True).astype("float32")
        spi     = srcs["SPI"].read(1, window=window, masked=True).astype("float32")
        flowacc = srcs["FlowAcc"].read(1, window=window, masked=True).astype("float32")
        aspect  = srcs["Aspect"].read(1, window=window, masked=True).astype("float32")

        # 有效像元：任何一个因子是 NoData 就不预测
        mask = (slope.mask | curv.mask | rough.mask | twi.mask |
                spi.mask | flowacc.mask | aspect.mask)

        out = np.full((window.height, window.width), profile["nodata"], dtype="float32")

        if (~mask).any():
            # ---- 变换：Aspect -> sin/cos ----
            asp = aspect.data.copy()
            asp[asp < 0] = 0.0  # ArcGIS 平坦区可能为 -1，处理成 0°
            asp_rad = np.deg2rad(asp)
            sin_asp = np.sin(asp_rad)
            cos_asp = np.cos(asp_rad)

            # ---- 变换：FlowAcc -> log1p ----
            fa = flowacc.data.copy()
            fa[fa < 0] = 0.0
            flowacc_log = np.log1p(fa)

            # ---- 变换：SPI -> log(平移后) ----
            spi_log = np.log(spi.data + shift + 1.0)

            # ★ 重要：特征顺序必须与训练一致！
            # 你截图里的训练重要性列表顺序是：
            # Slope, Curvatu, spi_log, sin_aspect, Roughness, flowacc_log, cos_aspect, TWI
            X = np.stack([
                slope.data,       # Slope
                curv.data,        # Curvatu
                spi_log,          # spi_log
                sin_asp,          # sin_aspect
                rough.data,       # Roughnes
                flowacc_log,      # flowacc_log
                cos_asp,          # cos_aspect
                twi.data          # TWI
            ], axis=-1)

            Xv = X[~mask].reshape(-1, X.shape[-1])
            pv = rf.predict_proba(Xv)[:, 1].astype("float32")
            out[~mask] = pv

        dst.write(out, 1, window=window)

for s in srcs.values():
    s.close()

print("✅ 输出完成:", OUT_PATH)
