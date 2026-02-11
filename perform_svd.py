import matplotlib
matplotlib.use('TkAgg') # 修复 PyCharm 弹窗报错

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ================= 1. 加载数据 =================
data_path = './dataset/mock_data/mock_casting_data.npy'
print(f"📂 正在加载数据: {data_path}")
data_matrix = np.load(data_path)

# data_matrix 的形状是 [200, 50, 50] -> [时间, 高, 宽]
n_time, height, width = data_matrix.shape
print(f"   数据形状: {n_time} (时间步) x {height*width} (空间网格)")

# ================= 2. 数据变形 (Flatten) =================
# 为了做 SVD，必须把每一张图 (50x50) 拉直成一个长条向量 (2500)
# 变形后矩阵 X 的形状: [2500, 200] -> [空间特征, 时间快照]
# 注意：通常 SVD 把空间放在行，时间放在列
X = data_matrix.reshape(n_time, -1).T
print(f"   变形后矩阵 X 形状: {X.shape} (行=空间点, 列=时间步)")

# ================= 3. 执行 SVD 分解 =================
print("🚀 正在执行 SVD 分解 (这可能需要几秒钟)...")
# X = U * Sigma * Vt
U, S, Vt = np.linalg.svd(X, full_matrices=False)

print("✅ SVD 分解完成！")

# ================= 4. 提取关键信息 =================
# 我们只保留前 10 个最重要的模态 (K=10)
# 既然原来的正弦波只是 1 维，现在的流场可能有多个维度的变化
K = 5

# 提取时间系数 (Temporal Coefficients)
# 这些系数就是 Transformer 接下来要预测的东西！
# 公式：Coeffs = Sigma * Vt
temporal_coeffs = np.diag(S) @ Vt
temporal_coeffs = temporal_coeffs[:K, :].T  # 取前K个，转置为 [Time, Features]

print(f"   提取了前 {K} 个时间模态，形状: {temporal_coeffs.shape}")

# ================= 5. 保存时间系数 (给 Transformer 用) =================
save_dir = './dataset/process_data/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df_coeffs = pd.DataFrame(temporal_coeffs, columns=[f'Mode_{i}' for i in range(K)])
# 添加时间列 (Date)，为了让 Transformer 的代码能读懂，我们需要伪造一个时间列
df_coeffs['date'] = pd.date_range(start='2024-01-01', periods=n_time, freq='h')
# 把 date 列放到第一列
cols = ['date'] + [c for c in df_coeffs.columns if c != 'date']
df_coeffs = df_coeffs[cols]

csv_path = os.path.join(save_dir, 'svd_coeffs.csv')
df_coeffs.to_csv(csv_path, index=False)
print(f"💾 时间系数已保存至: {csv_path} (这就是 Transformer 的输入！)")

# ================= 6. 可视化检查 =================
plt.figure(figsize=(12, 6))

# 画第1个最重要的模态随时间的变化
plt.subplot(2, 1, 1)
plt.plot(temporal_coeffs[:, 0], 'r-', label='Mode 0 (Most Energy)')
plt.title(f'Top 1 Temporal Coefficient (Representation of Heat Source Movement)')
plt.legend()
plt.grid(True)

# 画前5个模态
plt.subplot(2, 1, 2)
for i in range(K):
    plt.plot(temporal_coeffs[:, i], label=f'Mode {i}')
plt.title(f'Top {K} Temporal Coefficients')
plt.legend(loc='right')
plt.grid(True)

plt.tight_layout()
plt.show()