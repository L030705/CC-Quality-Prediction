import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

print("=== SVD 降阶演示程序启动 ===")

# --- 1. 读取图片 ---
image_path = 'test.jpg'

# 检查图片在不在
if not os.path.exists(image_path):
    print(f"❌ 错误：在当前文件夹没找到 {image_path}")
    print("👉 请检查：1. 图片是否放在代码旁边？ 2. 图片名字是不是写对了？")
    input("按回车键退出...")
    exit()

print(f"✅ 找到图片：{image_path}，正在读取...")

try:
    # 打开图片并转为灰度 (L模式)，因为SVD处理的是二维矩阵
    img = Image.open(image_path).convert('L')
    img_mat = np.array(img)
    print(f"✅ 图片转换成功，矩阵大小：{img_mat.shape}")
except Exception as e:
    print(f"❌ 图片读取失败：{e}")
    exit()

# --- 2. SVD 分解 (核心步骤) ---
print("⚡ 正在执行 SVD 分解 (这一步相当于拆解数据)...")
# U=空间特征, s=能量大小, Vt=时间特征
U, s, Vt = np.linalg.svd(img_mat, full_matrices=False)

# --- 3. 降阶重构 (模拟 POD) ---
# 我们只保留前 10% 的特征，看看能不能还原
keep_ratio = 0.1
k = int(len(s) * keep_ratio) 
print(f"📉 正在降阶：只保留前 {k} 个模态 (Top 10%)...")

# 重构公式：只用前k个特征乘回去
reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# --- 4. 保存结果图 ---
print("🎨 正在绘制对比图...")
plt.figure(figsize=(12, 6))

# 左边放原图
plt.subplot(1, 2, 1)
plt.imshow(img_mat, cmap='gray')
plt.title("Original Image (Ground Truth)")
plt.axis('off')

# 右边放降阶后的图
plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title(f"POD Reconstructed (Top {k} Modes)\nEnergy Preserved: {np.sum(s[:k]**2)/np.sum(s**2):.2%}")
plt.axis('off')

# 保存文件到当前目录
save_name = 'svd_result.png'
plt.savefig(save_name)
print(f"🎉 成功！结果图已保存为：{save_name}")
print("👉 快去文件夹里看看这张图吧！")

# 尝试弹窗显示
plt.show()