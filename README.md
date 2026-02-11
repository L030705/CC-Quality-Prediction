# Digital Twin Prototype: SVD + Transformer for Temperature Field Prediction
# 数字孪生原型系统：基于SVD与Transformer的连铸温度场时空预测

## 1. 项目简介 (Introduction)
本项目针对高维时空物理场数据（连铸温度场），提出了一套 **"仿真数据生成 -> SVD特征降维 -> Transformer时序预测 -> 物理场重构"** 的轻量化预测方案。通过提取物理场的主导模态（POD），成功将计算维度降低 99%，并利用 Transformer 实现了对未来温度场的精准预测。

## 2. 环境安装 (Installation)
本项目基于 Python 3.10 + PyTorch 构建。
```bash
pip install -r requirements.txt
