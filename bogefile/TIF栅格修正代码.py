import rasterio
from rasterio.merge import merge
import numpy as np
import os
import glob
from pathlib import Path


def aggressive_clean_dem_tif(input_path, output_path, new_nodata=-9999.0):
    """
    非常激进的DEM清洗函数，专门解决拼接后出现3.4e+38 / -3.4e+38的问题
    """
    print(f"\n正在激进清洗: {os.path.basename(input_path)}")

    try:
        with rasterio.open(input_path) as src:
            # 强制读取为float32
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            # 构建无效值掩码（越严格越好，DEM现实范围通常-500~9000m）
            invalid_mask = (
                    (data <= -999) |  # 异常负值
                    (data >= 9999) |  # 异常极大值（含3.4e38系列）
                    (data == 0) |  # 常见黑边/无效填充
                    np.isnan(data)  # NaN
            )

            # 如果原文件有nodata，也纳入无效范围
            if src.nodata is not None:
                invalid_mask |= (data == src.nodata)

            # 替换所有无效值为我们可控的nodata
            data[invalid_mask] = new_nodata

            # 更新profile（关键！）
            profile.update(
                dtype=rasterio.float32,
                nodata=new_nodata,
                compress='lzw',  # 压缩，节省空间
                tiled=True,
                blockxsize=512,
                blockysize=512
            )

            # 写入新文件
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)

        print(f"  → 清洗完成！输出: {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"  × 清洗失败: {input_path}")
        print(f"    错误: {str(e)}")
        return False


def mosaic_cleaned_dems(input_folder, output_mosaic_path, new_nodata=-9999.0):
    """
    步骤2：使用所有清洗后的tif进行拼接（方法A：rasterio.merge）
    """
    # 查找所有清洗后的文件（建议放在 cleaned/ 子文件夹）
    cleaned_files = glob.glob(os.path.join(input_folder, "*_clean.tif"))

    if not cleaned_files:
        print("错误：没有找到任何 *_clean.tif 文件！请先运行清洗步骤。")
        return

    print(f"\n找到 {len(cleaned_files)} 个已清洗文件，开始拼接...")
    print("文件列表：")
    for f in cleaned_files:
        print(f"  - {os.path.basename(f)}")

    try:
        # 读取第一个文件作为参考元数据
        with rasterio.open(cleaned_files[0]) as src:
            meta = src.meta.copy()

        # 执行合并，**明确指定nodata**（最关键一步！）
        mosaic_array, out_transform = merge(
            cleaned_files,
            nodata=new_nodata,  # 必须和清洗时一致
            method='first'  # 'first'：保留第一幅的值；也可改'last'
        )

        # 更新最终输出元数据
        meta.update({
            "driver": "GTiff",
            "height": mosaic_array.shape[1],
            "width": mosaic_array.shape[2],
            "transform": out_transform,
            "nodata": new_nodata,
            "compress": "lzw",
            "dtype": "float32",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })

        # 写入最终镶嵌结果
        with rasterio.open(output_mosaic_path, "w", **meta) as dest:
            dest.write(mosaic_array)

        print(f"\n拼接完成！最终结果保存至：")
        print(f"  {output_mosaic_path}")
        print(f"输出尺寸：{mosaic_array.shape[2]} × {mosaic_array.shape[1]}")

    except Exception as e:
        print(f"拼接失败：{str(e)}")


# =======================
#       主程序使用示例
# =======================
if __name__ == "__main__":
    # ========================
    #  修改这里为你自己的路径！
    # ========================
    INPUT_FOLDER = r"C:\Users\iniss\Desktop\mission\processing_tif\MERGED"  # 原始tif所在文件夹
    CLEANED_FOLDER = r"C:\Users\iniss\Desktop\mission\processing_tif\MERGED\clearnd"  # 清洗后存放文件夹（建议新建）
    FINAL_MOSAIC = r"C:\Users\iniss\Desktop\mission\processing_tif\MERGED\merged_new\AA.tif"  # 最终拼接输出路径

    # 确保输出文件夹存在
    os.makedirs(CLEANED_FOLDER, exist_ok=True)

    # 步骤1：清洗所有原始tif
    print("=== 步骤1：开始批量清洗原始DEM文件 ===\n")

    original_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))

    if not original_files:
        print("错误：在指定文件夹中没有找到任何 .tif 文件！")
    else:
        success_count = 0
        for idx, orig_file in enumerate(original_files, 1):
            print(f"[{idx}/{len(original_files)}]")
            base_name = Path(orig_file).stem
            clean_path = os.path.join(CLEANED_FOLDER, f"{base_name}_clean.tif")

            if aggressive_clean_dem_tif(orig_file, clean_path):
                success_count += 1

        print(f"\n清洗阶段完成：成功 {success_count}/{len(original_files)} 个文件")

        # 步骤2：拼接
        if success_count > 0:
            print("\n=== 步骤2：开始拼接已清洗文件 ===\n")
            mosaic_cleaned_dems(CLEANED_FOLDER, FINAL_MOSAIC)
        else:
            print("没有成功清洗的文件，跳过拼接。")