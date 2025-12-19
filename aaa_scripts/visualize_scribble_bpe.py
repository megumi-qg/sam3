import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools import mask as mask_util

def visualize_scribbles(json_path, image_dir, image_id_to_show=0):
    """
    解码 RLE mask 并叠加在原图上进行可视化
    """
    # 1. 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 获取图片信息
    img_info = next((item for item in data['images'] if item['id'] == image_id_to_show), None)
    if img_info is None:
        print(f"未找到 ID 为 {image_id_to_show} 的图片")
        return

    print(f"正在处理图片: {img_info['file_name']}")
    
    # 3. 读取原图
    # 注意：这里需要拼接你的实际图片路径
    img_path = os.path.join(image_dir, img_info['file_name'])
    
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"警告：找不到图片文件 {img_path}，将使用全黑背景代替以展示 Mask 形状。")
        image = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)

    # 4. 获取该图片的所有标注
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id_to_show]
    
    # 建立类别 ID 到颜色的映射 (R, G, B)
    # 假设类别 1:红, 2:绿, 3:蓝
    category_colors = {
        1: (255, 0, 0),   # Right Ventricle
        2: (0, 255, 0),   # Myocardium
        3: (0, 0, 255)    # Left Ventricle
    }

    # 创建一个用于叠加的 Mask 层
    overlay = image.copy()
    alpha = 0.6  # 透明度

    print(f"找到 {len(annotations)} 个标注片段。")

    for ann in annotations:
        rle_obj = ann['segmentation']
        
        # 5. 核心步骤：解码 RLE
        # pycocotools 的 decode 需要 count 是字节串或字符串，通常 JSON 读出来是字符串
        # 如果报错，可能需要把 counts 转为 bytes: rle_obj['counts'] = rle_obj['counts'].encode('utf-8')
        binary_mask = mask_util.decode(rle_obj)
        
        # binary_mask 现在是一个 0/1 的二维 numpy 数组 (height, width)
        
        cat_id = ann['category_id']
        color = category_colors.get(cat_id, (255, 255, 255)) # 默认白色
        
        # 将 Mask 区域上色
        # binary_mask == 1 的位置涂上对应颜色
        for c in range(3): # R, G, B
            overlay[:, :, c] = np.where(
                binary_mask == 1,
                color[c],
                overlay[:, :, c]
            )

    # 6. 混合原图和 Mask
    result_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 7. 绘图
    plt.figure(figsize=(10, 8))
    plt.title(f"Image ID: {image_id_to_show} | RLE Decode Check")
    plt.imshow(result_image)
    plt.axis('off')
    plt.savefig('scribble_visualization.png')
    # plt.show()

# ==========================================
# 使用示例
# ==========================================

# 1. 将你的 JSON 文件路径填在这里
json_file = '/home/gaoqi/sam3/dataset/ACDC_new/train/image_scribble_annotations.coco.json' 

# 2. 将你的图片根目录填在这里 (对应 JSON 中 file_name 的相对路径)
# 如果 JSON 里是 "images/patient..."，这里填包含 "images" 文件夹的父目录
img_root_dir = '/home/gaoqi/sam3/dataset/ACDC_new/train' 

# 运行 (检查 ID 为 0 的图片)
# 确保你已经创建了 JSON 文件，或者把代码放在有 JSON 的目录下
if __name__ == "__main__":
    # 为了演示，如果文件不存在，我先创建一个假的 JSON 文件用于测试代码逻辑
    if not os.path.exists(json_file):
        print("未找到 JSON文件，正在生成示例文件以供测试代码逻辑...")
        sample_data = {
          "images": [{"id": 0, "file_name": "images/patient001_frame01_slice001.png", "height": 256, "width": 216}],
          "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "segmentation": {"size": [256, 216], "counts": "mk>2m72L5M20I\\HOY?OcH2N2O100000001O0XIJe54YJNh5MZJ4h6100O10LXH0[7NQI0E2Z71PIOR74iHMZ7;1N3LZ[P1"}},
            {"id": 1, "image_id": 0, "category_id": 2, "segmentation": {"size": [256, 216], "counts": "]\\d03l721N1N]o82bPG000010NU\\f0"}}
          ],
          "categories": [{"id": 1, "name": "RV"}, {"id": 2, "name": "Myo"}]
        }
        with open(json_file, 'w') as f:
            json.dump(sample_data, f)
            
    visualize_scribbles(json_file, img_root_dir, image_id_to_show=200)