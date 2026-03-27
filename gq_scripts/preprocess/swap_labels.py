# 交换 NIfTI 文件中标签 0 和 5，保存到新文件
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    print("请先安装 nibabel: pip install nibabel")
    exit(1)

path = "/home/gaoqi/dataset/using/btcv_1/train/scribble_bench/0759564-Mask.nii.gz"
path = Path(path)

# 新文件：同一目录下，文件名加 _swapped，如 0759564-Mask_swapped.nii.gz
out_path = path.parent / (path.stem.replace(".nii", "") + "_swapped.nii.gz")

img = nib.load(str(path))
arr = np.asarray(img.dataobj).copy()
# 交换 0 与 5：必须按“原数组读、新数组写”，否则先改 0→5 再改 5→0 会全变成 0
out = arr.copy()
out[arr == 0] = 5
out[arr == 5] = 0
out_img = nib.Nifti1Image(out, img.affine, img.header)
nib.save(out_img, str(out_path))
print(f"已完成：标签 0 与 5 已互换，已保存到新文件 {out_path}")
