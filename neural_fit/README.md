# 用 2→10→1 神经网络拟合二次函数

目标：`z = x² + y² + 10`，默认拟合范围为 `x,y ∈ [-5,5]`。

网络使用 PyTorch：输入层 2 个节点，隐藏层 10 个节点（Tanh 激活），输出层 1 个节点（线性输出），共 41 个可训练参数。输入直接使用 x、y，没有预先提供平方特征。输入和输出按训练数据进行缩放，预测时自动还原到 z 的原始单位。

## 配置和运行（Python 3.12，CPU 即可）

在仓库根目录执行：

```bash
cd neural_fit
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/python train.py --predict 3 4
```

依赖固定版本的 CPU 版 PyTorch、NumPy 和 Matplotlib，无需 GPU。

本次云端已配置的环境仍在 `/workspace/neural_fit/.venv`，可以直接运行仓库代码：

```bash
/workspace/neural_fit/.venv/bin/python /workspace/test/neural_fit/train.py --predict 3 4
```

## 训练与结果

固定随机种子 42，独立采样 4096 个训练点、1024 个验证点、2048 个测试点。先执行 2000 步 Adam，再使用 L-BFGS 优化。用验证集选择模型，最后才计算测试集指标。

`outputs/` 保存：

- `model.pt`：网络权重和缩放参数。
- `metrics.json`：测试集 MSE、RMSE、MAE、R² 与预测例子。
- `fit.png`：真实曲面、预测曲面、网格误差和训练曲线。
- `test_predictions.csv`：所有测试点的真实值、预测值和误差。
- `loss.csv`：训练及验证损失，单位为原始 z 的平方。

修改范围和 Adam 步数，例如：

```bash
.venv/bin/python train.py --radius 10 --epochs 3000 --output outputs_range10
.venv/bin/python train.py --output outputs_range10 --predict 3 4
```

神经网络学习的是指定范围内的近似；超出训练范围时，预测准确度不受保证。L-BFGS 在每次调用中可执行多个内部迭代，因此图中横轴是优化器调用次数。
