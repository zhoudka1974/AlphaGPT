import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multivariate time series anomaly detection.
    
    Architecture:
        Encoder: Input -> LSTM -> Hidden State (bottleneck)
        Decoder: Hidden State -> Repeat -> LSTM -> Reconstructed Output
    
    Args:
        input_dim (int): Number of features (e.g., 1 for univariate, 5 for multivariate)
        hidden_dim (int): Hidden size of LSTM (bottleneck dimension)
        num_layers (int): Number of LSTM layers (default: 1)
        dropout (float): Dropout rate (default: 0.0)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: compress time series into a latent vector
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Decoder: reconstruct the sequence from latent vector
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection: map decoder output to original feature space
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reconstruct input sequence.
        
        Args:
            x (torch.Tensor): Input time series of shape [B, T, D]
                B = batch size
                T = sequence length
                D = number of features (input_dim)
                
        Returns:
            recon_x (torch.Tensor): Reconstructed sequence of shape [B, T, D]
        """
        B, T, D = x.shape
        
        # Encode: get final hidden state
        _, (h_n, c_n) = self.encoder(x)  # h_n: [num_layers, B, hidden_dim]
        
        # Use last layer's hidden state as bottleneck
        latent = h_n[-1]  # [B, hidden_dim]
        
        # Repeat latent vector to match sequence length
        decoder_input = latent.unsqueeze(1).repeat(1, T, 1)  # [B, T, hidden_dim]
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        
        # Project to original feature space
        recon_x = self.fc_out(decoder_output)  # [B, T, D]
        
        return recon_x
    
import torch
import numpy as np

# Simulate 1000 sequences of 50-timestep CPU usage (normal + some anomalies)
np.random.seed(42)
#生成一个模拟的“正常”时间序列数据集,950条时间序列.每一个序列50个时间步.每一个时间步有一个变量.共47500个浮点数
# 从均值0.5,标准差0.1的正太分布中随机采样
# [ 
#  seq0_t0_f0, seq0_t1_f0, ..., seq0_t49_f0,
#  seq1_t0_f0, seq1_t1_f0, ..., seq1_t49_f0,
#  ...
#  seq949_t0_f0, ..., seq949_t49_f0
# ]
# seqX 表示第 X 条时间序列（共 950 条）
# tY 表示第 Y 个时间步（0 到 49）
# f0 表示第 0 个特征（因为只有 1 个特征）
normal_data = np.random.normal(loc=0.5, scale=0.1, size=(950, 50, 1)) 
anomaly_data = np.random.normal(loc=0.9, scale=0.2, size=(50, 50, 1))  # high CPU
# normal_data:   [样本0, 样本1, ..., 样本949]     → 950 个样本
# anomaly_data:  [样本0, 样本1, ..., 样本49]      → 50 个样本
# ------------------------------------------------------------
# data:          [样本0～949 (正常), 样本0～49 (异常)] → 1000 个样本
# 这是个拼接动作,np.concatenate([normal_data, anomaly_data], axis=0) 时，除了拼接轴（axis=0）之外，其他所有维度的大小必须完全一致。
# 合并两个NumPy 数组,类型为NumPy ndarray . 形状(1000,50,1), 数据类型float64. 存储在RAM中,由NumPy管理
data = np.concatenate([normal_data, anomaly_data], axis=0)
# 生成一个torch.tensor , 类型为torch.float32 , 保存在cpu上.
data = torch.tensor(data, dtype=torch.float32)

# Create model
# 创建一个用于时间序列异常检测的 LSTM 自编码器模型实例. 
# 他的forward 先encode , 然后decode , 折腾啥?
model = LSTMAutoencoder(input_dim=1, hidden_dim=32)
# 定义一个均方误差（MSE）损失函数，并设置其不进行聚合（即保留每个元素的损失值）
criterion = nn.MSELoss(reduction='none')  # per-timestep loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop (unsupervised)
    # 将模型设为训练模式。
    # 影响哪些层？
            # 虽然当前模型没有 Dropout 或 BatchNorm，但这是标准做法。
            # 如果未来加入这些层，train() 会启用 dropout 和 batch 统计。
    #对比：评估时需调用 model.eval()。
model.train()
    # 进行 50 轮（epochs）训练。
    # epoch 含义：遍历整个数据集一次为一个 epoch。
    # 为什么 50？
        # 对于简单合成数据，50 轮通常足够收敛。
        # 实际项目中可能需要更多（如 100～500），或配合早停（early stopping）。
for epoch in range(50):
        # 功能：将所有参数的梯度清零。
        # 么需要？
            # Torch 默认累加梯度（方便 RNN 等场景）。
            # 反向传播前必须清零，否则梯度会错误累积。
        # 写法：model.zero_grad()，但推荐用 optimizer.zero_grad()    
    optimizer.zero_grad()
        # 将数据 data 输入模型，得到重建结果 recon。
        # 输入/输出形状：
            # data: [1000, 50, 1]（1000 条序列，每条 50 步，单变量）
            # recon: [1000, 50, 1]（重建后的序列）
        # 注意：
            # 虽然 data 包含正常+异常样本，但自编码器在无监督训练中通常只用正常数据。
            # 此处为了简化演示用了全部数据，实际应只用 normal_data 训练！    
    recon = model(data)
        # 计算重建误差，并取平均作为最终损失
        # 分步解析：
            # criterion(recon, data) → 返回 [1000, 50, 1] 的逐元素 MSE
            # .mean() → 对所有元素求平均，得到标量 loss
    loss = criterion(recon, data).mean()
        # 自动计算损失对所有参数的梯度。
    loss.backward()
        # 根据当前梯度更新模型参数。
        # 内部操作（以 Adam 为例）：
            # 计算一阶/二阶动量
            # 调整学习率
            # 执行参数更新：param = param - lr * grad
         # 🔁 完成一次“前向 → 损失 → 反向 → 更新”闭环。        
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        # 整体流程总结
        # 表格
        # 步骤	代码	目的
        # . 初始化	model, criterion, optimizer	构建训练组件
        # . 训练模式	model.train()	启用训练行为
        # . 循环	for epoch in ...	多轮优化
        # . 清梯度	optimizer.zero_grad()	避免梯度累积
        # . 前向	recon = model(data)	获取重建结果
        # . 损失	loss = ... .mean()	量化重建误差
        # . 反向	loss.backward()	计算梯度
        # . 更新	optimizer.step()	优化参数
        # . 日志	print(...)	监控训练



# Anomaly scoring: reconstruction error per sequence
    # 将模型设为评估（推理）模式。
    # 作用：
        # 如果模型包含 Dropout、BatchNorm 等层，会禁用它们的训练行为（如 dropout 随机失活、BN 使用 batch 统计）。
        # 虽然当前 LSTMAutoencoder 没有这些层，但这是标准且必要的做法。
    # 对比：训练时用 model.train()，评估时必须用 model.eval()。
model.eval()
    # 在该代码块内不计算或存储梯度。
    # 目的：
        # 节省内存：避免构建计算图（computation graph）
        # 加速推理：跳过自动微分的开销
        # 防止意外更新：确保模型参数不会被修改
    # 适用场景：所有推理（inference）或评估（evaluation） 阶段。
        #⚠️ 在训练循环外做预测时，务必使用 torch.no_grad()。
with torch.no_grad():
    # 将整个数据集 data（含正常+异常）输入模型，得到重建序列。
    # 输入/输出形状：
        # data: [1000, 50, 1]（950 正常 + 50 异常）
        # recon: [1000, 50, 1]（模型对每条序列的重建）
    # 关键假设：
        # 模型只在正常数据上训练过 → 能很好重建正常序列
        # 对异常序列重建效果差 → 重建误差大
    # 📌 注意：这里 data 是混合数据，用于测试模型泛化能力。    
    recon = model(data)
    # 计算每条时间序列的平均平方重建误差。
    # 分步解析：
        # (recon - data) → 逐元素误差，shape [1000, 50, 1]
        # ** 2 → 平方误差（等价于 MSE 的分子）
        # torch.mean(..., dim=[1, 2]) → 对 时间步（dim=1）和特征（dim=2）求平均
    # 结果：
        # seq_errors 是一个长度为 1000 的一维张量
        # seq_errors[i] 表示第 i 条序列的整体重建误差
    # ✅ 这就是异常评分（anomaly score）：误差越大，越可能是异常！    
    seq_errors = torch.mean((recon - data) ** 2, dim=[1, 2])  # [1000]

# Threshold: top 5% as anomalies
threshold = torch.quantile(seq_errors, 0.95)
anomaly_pred = seq_errors > threshold

print(f"Detected {anomaly_pred.sum().item()} anomalies (expected ～50)")