import torch
"""
    用于时间序列信号处理或自动特征工程(如量化因子挖掘、遗传编程)的原子操作(primitive operations),
    并利用 PyTorch 的 torch.jit.script 编译优化以提升运行效率。
    整体设计常见于自动化 alpha 挖掘、程序合成或神经符号系统中。
"""
@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """
        功能:时间序列延迟(lag)操作
        返回 x 向后移动 d 期的结果(即 x[t-d] 对应输出 t 时刻)
        边界处理:前 d 个时间步用 0 填充(因果 padding,无未来信息泄露)
    """
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    result = torch.cat([pad, x[:, :-d]], dim=1)
    # import ipdb ; ipdb.set_trace()
    return result

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
        功能:条件选择门(类似 if-else)
        - 若 condition > 0,输出 x；否则输出 y
        - 向量化实现:避免 Python 控制流,支持 GPU 并行
        数学等价于:
        output={
                 x if condition>0
                 y otherwise
                }
        ✅ 用于构建条件逻辑(如“如果波动率高,则用保守策略”)
    """
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """
        功能:检测“跳跃”或极端异常值
        计算每行(每个代币)的 Z-score:   z=(x - μ)/σ
        若   z>3 (即偏离均值 3 个标准差以上),输出 z - 3 ；否则输出 0
        意义:识别价格/指标的突发性暴涨暴跌(meme 币常见)
        📌 输出为非负值,越大表示“跳得越猛”
    """    
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    """
    功能:指数衰减加权(近似)
        当前值权重 1.0,昨日 0.8,前日 0.6
        虽非严格指数衰减(如 EWMA),但实现了“近期更重要”的思想
        用途:平滑信号、赋予近期数据更高优先级
        💡 可视为一个简单的 FIR 滤波器
    """
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

OPS_CONFIG = [
    #    这是一个 操作原语库(primitive set),每个元素是三元组 (name, function, arity):
    #    字段	含义
    #    name	操作名称(字符串标识)
    #    function	可调用的函数(lambda 或 JIT 函数)
    #    arity	所需输入参数个数(1=一元,2=二元,3=三元)
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1)
]