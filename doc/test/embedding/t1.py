import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块：为输入序列添加位置信息
    使用正弦和余弦函数生成位置编码
    """

    def __init__(self, max_pos: int, embed_dim: int):
        super(PositionalEncoding, self).__init__()
        
        # 初始化位置编码数组 PE，形状为 (max_pos, embed_dim)
        PE = torch.zeros(max_pos, embed_dim)  # 存储最终的位置编码
        
        # 生成从0到max_pos-1的位置索引，形状为 (max_pos, 1)
        pos = torch.arange(0, max_pos).unsqueeze(1).float()  # [max_pos, 1]
        
        # 生成维度索引，只取偶数维度（用于计算 sin/cos 的分母）
        multi_term = torch.arange(0, embed_dim, 2).float()  # [embed_dim//2]

        # 计算分母项：10000^(2i/dim)，其中 i 是维度索引
        # 等价于 exp(-log(10000) * i / d_model)
        multi_term = torch.exp(multi_term * (-math.log(10000.0) / embed_dim))

        # 正弦部分：PE[pos, 2i] = sin(pos / 10000^(2i/d))
        PE[:, 0::2] = torch.sin(pos * multi_term)

        # 余弦部分：PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
        PE[:, 1::2] = torch.cos(pos * multi_term)

        # 将PE注册为一个不参与梯度更新的缓冲区（常量）
        self.register_buffer('PE', PE.unsqueeze(0))  # 形状变为 (1, max_pos, embed_dim)
        # self.PE = PE

    def forward(self, x):
        """
        前向传播:将位置编码加到输入张量x上
        x: 输入张量，形状 (batch_size, seq_len, embed_dim)
        """
        # 取出对应长度的位置编码，并与输入相加
        return x + self.PE[:, :x.size(1)].clone().detach()

if __name__ == "__main__":
    max_pos = 10      # 最大序列长度
    embed_dim = 4     # 词向量维度

    # 定义位置编码模型
    model = PositionalEncoding(max_pos, embed_dim)

    # 输入数据：2个样本，每个样本长度为max_pos = 10，每个维度是embed_dim=4
    x = torch.zeros(2, 5, embed_dim)  # 形状: (batch_size, seq_len, embed_dim)

    # 将x传入模型，计算添加位置信息的结果output
    output = model(x)

    # 打印结果
    print("x:")
    print(x)
    print("PE:")
    print(model.PE)
    print("output:")
    print(output)