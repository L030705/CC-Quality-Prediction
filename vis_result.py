import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 1. 自动寻找最新的预测结果 =================
# Transformer 跑完的结果都藏在 results 文件夹里
result_dir = './results/'
# 找最新的那个文件夹 (名字最长的那个 usually)
list_dirs = glob.glob(result_dir + 'long_term_forecast_*')
if not list_dirs:
    print("❌ 没找到预测结果！请确认你刚才运行了 run.py 并没有报错。")
    exit()

latest_dir = max(list_dirs, key=os.path.getmtime)
print(f"📂 锁定最新的实验结果: {latest_dir}")

# 加载 AI 预测的数据 (pred.npy) 和 真实数据 (true.npy)
preds = np.load(os.path.join(latest_dir, 'pred.npy'))
trues = np.load(os.path.join(latest_dir, 'true.npy'))

print(f"   加载成功！预测数据形状: {preds.shape}")
# shape: [样本数, 预测长度24, 特征数5]

# ================= 2. 重新获取空间信息 (SVD) =================
# 因为我们之前只存了时间系数，没存空间基函数(U)，这里快速重新算一下
# 别担心，几千行数据对于 SVD 是一瞬间的事
print("🔄 正在重新提取空间特征 (用于图像还原)...")
data_matrix = np.load('./dataset/mock_data/mock_casting_data.npy')
n_time, height, width = data_matrix.shape
X = data_matrix.reshape(n_time, -1).T
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# 只需要前 5 个空间模态 (Spatial Modes)
# U_k 的形状: [2500, 5] -> 代表了 5 张基础的“热力图脸谱”
K = 5
U_k = U[:, :K]

print("✅ 空间特征提取完毕！开始还原...")

# ================= 3. 还原热力图 (见证奇迹) =================
# 我们随便挑一个样本来看看，比如第 10 个样本
sample_idx = 0
# 取出 AI 预测的 24 步未来的系数 [24, 5]
pred_coeffs = preds[sample_idx]
# 取出 真实的 24 步未来的系数 [24, 5]
true_coeffs = trues[sample_idx]

# 核心公式：图像 = 空间基(U) * 时间系数(Sigma*Vt)
# 这里我们的 preds 已经是 (Sigma*Vt) 了，所以直接乘 U 即可
# 矩阵乘法: [2500, 5] x [5, 24] = [2500, 24]
reconstructed_pred = U_k @ pred_coeffs.T
reconstructed_true = U_k @ true_coeffs.T

# 变回图片形状 [50, 50, 24]
rec_pred_imgs = reconstructed_pred.reshape(height, width, -1)
rec_true_imgs = reconstructed_true.reshape(height, width, -1)

# ================= 4. 画图对比 =================
# 我们看看预测的“第 24 步” (最后一步) 长啥样
step = 23  # index from 0

plt.figure(figsize=(12, 5))

# 真实的热力图
plt.subplot(1, 3, 1)
plt.imshow(rec_true_imgs[:, :, step], cmap='jet', vmin=0, vmax=1)
plt.title(f'Ground Truth (Future Step {step + 1})')
plt.colorbar(fraction=0.046, pad=0.04)

# AI 预测的热力图
plt.subplot(1, 3, 2)
plt.imshow(rec_pred_imgs[:, :, step], cmap='jet', vmin=0, vmax=1)
plt.title(f'Prediction (Transformer)')
plt.colorbar(fraction=0.046, pad=0.04)

# 误差图 (哪里预测错了？)
plt.subplot(1, 3, 3)
error_img = np.abs(rec_true_imgs[:, :, step] - rec_pred_imgs[:, :, step])
plt.imshow(error_img, cmap='Purples', vmin=0, vmax=0.2)
plt.title(f'Prediction Error (Difference)')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

print("🎉 恭喜！数字孪生全流程跑通！")