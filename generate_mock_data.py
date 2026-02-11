import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ================= 配置参数 =================
frames = 2000  # 总时间步数 (模拟2000秒)
height = 50  # 断面高度 (网格数)
width = 50  # 断面宽度 (网格数)
save_path = './dataset/mock_data/'  # 数据保存路径

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"🚀 开始生成仿真数据...")
print(f"   尺寸: {frames} (时间) x {height} (高) x {width} (宽)")

# ================= 核心物理模拟 =================
# 初始化数据矩阵 [时间, 高, 宽]
data_matrix = np.zeros((frames, height, width))

# 模拟一个移动的热源 (高斯分布)
# 热源中心 (cx, cy) 随时间 t 移动
x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)

for t in range(frames):
    # 1. 设定热源中心位置：随时间从 (10,10) 移动到 (40,40)
    # 模拟连铸过程中的拉坯移动或温度场漂移
    center_x = 10 + (t / frames) * 30
    center_y = 10 + (t / frames) * 30

    # 2. 设定热源强度：随时间慢慢冷却 (从 1.0 降到 0.5)
    intensity = 1.0 * np.exp(-0.005 * t)

    # 3. 计算高斯热场 (二维正态分布公式)
    # (X-cx)^2 + (Y-cy)^2 决定了距离中心的远近
    sigma = 5.0  # 热源扩散范围
    heat_field = intensity * np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma ** 2))

    # 4. 加入一点点随机噪声 (模拟真实工况的传感器波动)
    noise = 0.05 * np.random.randn(height, width)

    # 5. 合成最终温度场
    data_matrix[t, :, :] = heat_field + noise

print("✅ 数据生成完毕！")

# ================= 保存数据 =================
# 保存为 .npy 格式，供下一步 POD 使用
file_name = os.path.join(save_path, 'mock_casting_data.npy')
np.save(file_name, data_matrix)
print(f"💾 数据已保存至: {file_name}")

# ================= 可视化验证 (生成动图) =================
print("🎥 正在生成预览动画，请稍候...")
fig = plt.figure()
ims = []
for i in range(0, frames, 2):  # 每隔2帧画一次，快一点
    im = plt.imshow(data_matrix[i], animated=True, cmap='jet', vmin=0, vmax=1)
    title = plt.text(0.5, 1.01, f'Time Step: {i}', ha="center", va="bottom",
                     transform=plt.gca().transAxes, fontsize="large")
    ims.append([im, title])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
plt.title("Simulated Moving Heat Source (Continuous Casting)")
plt.colorbar(label='Temperature')

plt.show()
print("🎉 第一步完成！请查看弹出的动图。")