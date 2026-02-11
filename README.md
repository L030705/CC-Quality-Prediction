# Digital Twin Prototype: SVD + Transformer for Temperature Field Prediction
# 数字孪生原型系统：基于SVD与Transformer的连铸温度场时空预测

## 1. 项目简介 (Introduction)
本项目针对高维时空物理场数据（连铸温度场），提出了一套 **"仿真数据生成 -> SVD特征降维 -> Transformer时序预测 -> 物理场重构"** 的轻量化预测方案。通过提取物理场的主导模态（POD），成功将计算维度降低 99%，并利用 Transformer 实现了对未来温度场的精准预测。

## 2. 环境安装 (Installation)
本项目基于 Python 3.10 + PyTorch 构建。
```bash
pip install -r requirements.txt
```
## 3. 快速复现（Quick Start）
请按顺序执行以下脚本即可复现全流程实验。
第一步：生成仿真数据
构建包含“移动热源”与“冷却衰减”特征的二维温度场数据 (2000x50x50)。
```bash
python generate_mock_data.py
# 输出：dataset/mock_data/mock_casting_data.npy
```
第二步：SVD 特征降维
提取温度场的前 5 阶时间模态系数。
```bash
python perform_svd.py
# 输出：dataset/process_data/svd_coeffs.csv
```
第三步：运行 Transformer 模型
利用提取的时间系数训练模型，并预测未来 24 个时间步的变化。
```bash
python run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/process_data/ --data_path svd_coeffs.csv --model_id svd_model --model Transformer --data custom --features M --seq_len 48 --label_len 24 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 5 --dec_in 5 --c_out 5 --des 'Exp' --itr 1 --train_epochs 5 --batch_size 16 --target Mode_0
```
第四步：可视化还原 (Digital Twin Visualization)
将预测的抽象系数反投影回二维空间，生成对比云图。
```bash
python vis_result.py
```
## 4. 实验结果 (Results)
数字孪生效果对比: 模型成功预测了热源的移动路径与强度衰减，误差图 (Error Map) 接近全零。
