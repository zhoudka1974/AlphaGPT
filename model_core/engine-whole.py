import torch
import json
import ipdb
import os
import pandas as pd
import torch.nn as nn
import sqlalchemy
import torch.nn.functional as F

from torch.distributions import Categorical
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# config.py
class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 8192
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = 6

#ops.py
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

# factors.py
class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization
        RMS 归一化层
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        """
            衡量流动性健康度 —— 流动性池深度相对于完全稀释市值(FDV)的比例。
            逻辑:
                FDV 越高,项目“理论市值”越大
                若流动性远小于 FDV(如 FDV= 1B,流动性= 10k),则极易被砸盘(滑点巨大)
            归一化:
                ratio * 4.0:假设理想 ratio ≈ 0.25(即流动性占 FDV 的 25%),此时输出为 1.0
                clamp(0, 1):限制在 [0,1] 区间,0=极差,1=极好
                ✅ 输出可作为“抗砸盘能力”评分
            """
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        """
            衡量单根K线的买卖力量不平衡程度(类似"实体占比")。
            计算:
                body = close - open_:阳线为正,阴线为负
                range_hl = high - low:整根K线波动范围
                strength = body / range_hl:实体占总振幅的比例(∈ [-1, 1])
            非线性压缩:
                tanh(strength * 3):放大中等强度信号,抑制极端值(平滑)
            ✅ 正值表示买方强势,负值表示卖方强势
        """
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        """
            检测FOMO(错失恐惧)情绪的加速度 —— 成交量增长是否在加速。
            步骤:
                1, vol_chg:成交量环比变化率(昨日为分母,+1 防除零)
                2, acc = Δ(vol_chg):成交量变化率的二阶导数(加速度)
            意义:
                1, acc > 0:FOMO 在加剧(越来越多人冲入)
                2, acc < 0:FOMO 减退(热度下降)
            裁剪:clamp(-5, 5) 防止异常值干扰
            ⚠️ 注意:torch.roll 会导致首列使用最后一列数据(循环移位),实际应 padding。但此处可能接受边界误差。
        """
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        """
            衡量价格相对于移动平均线的偏离程度(判断是否“暴涨”或“超买”)。
            实现细节:
                手动 padding(前补 0)→ 计算从第 window 天开始的 MA
                unfold(1, window, 1):滑动窗口提取子序列
                dev = (price - MA) / MA:相对偏离率(类似布林带 z-score)
            用途:
                dev >> 0:价格远高于均线 → 可能过热
                dev << 0:价格远低于均线 → 可能超卖
            ✅ 典型“追涨杀跌”信号源
        """
        pad = torch.zeros((close.shape[0], window-1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns
            检测波动率聚集效应(金融经典现象:高波动后往往接高波动)。
            步骤:
                1, 计算对数收益率 ret = log(P_t / P_{t-1})
                2, 平方收益率 ret_sq → 代表瞬时波动率
                3, 对 ret_sq 做移动平均 → 得到局部波动率估计
                4, 开方 → 近似标准差(波动率)
            输出:滚动窗口内的历史波动率
            ✅ 用于识别“高波动期”,可能伴随 meme 币剧烈炒作
        """
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals
            检测动量反转信号 —— 上涨/下跌趋势是否突然转向。
            逻辑:
                1, mom:过去 window 天的累计收益率(动量方向)
                2, mom_prev:前一天的动量
                3, mom * mom_prev < 0:符号相反 → 发生动量反转
            输出:二值信号(1=反转,0=无反转)
            ✅ 用于捕捉“见顶回落”或“止跌反弹”的转折点
        """
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = torch.roll(mom, 1, dims=1)
        reversal = (mom * mom_prev < 0).float()
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection
            实现 RSI(相对强弱指数) 的变体,用于判断超买/超卖。
            标准 RSI 步骤:
                1, 计算每日涨跌(gains, losses)
                2, 计算 window 日平均涨幅/跌幅
                3, RS = avg_gain / avg_loss
                4, RSI = 100 - 100/(1+RS)
            归一化:
                (RSI - 50) / 50 → 将 [0,100] 映射到 [-1, 1]
                    -1:极端超卖
                    0:中性
                    +1:极端超买
            ✅ 经典震荡指标,适用于 meme 币的高波动环境        
        """
        ret = close - torch.roll(close, 1, dims=1)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Advanced feature engineering with multiple factor types
        从原始市场数据中构建一个12维的高级特征空间(feature space),特别针对高波动性资产(如 meme 币)设计。
        它结合了基础技术指标、行为金融信号和稳健归一化方法,
        输出可直接用于机器学习模型(如预测、分类或强化学习)的标准化特征张量.
    """
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def robust_norm(self, t):
        """Robust normalization using median absolute deviation
            稳健归一化(基于中位数和 MAD)
            对输入张量 t(shape: [num_tokens, time_steps])进行稳健标准化,避免受极端值(outliers)影响。
            步骤:
                1,计算每行(每个代币)的中位数 → 比均值更抗异常值。
                2, 计算 MAD(Median Absolute Deviation):
                    MAD = median(|Xi - median(X)|)
                3,标准化:(x - median) / MAD
            裁剪:限制在 [-5, 5],防止残余异常值干扰模型。
            ✅ 为什么用 MAD?
            Meme 币价格常出现暴涨暴跌(如 100x),均值和标准差会被严重扭曲。MAD 对异常值不敏感,更适合此类数据。            
        """
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)
    
    def compute_advanced_features(self, raw_dict):
        """Compute 12-dimensional feature space with advanced factors 构建12维特征
        """
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        # Basic factors 计算基础因子(Basic Factors)
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9)) # 对数收益率
        liq_score = MemeIndicators.liquidity_health(liq, fdv)   # 流动性健康度
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l) # 买卖压力
        fomo = MemeIndicators.fomo_acceleration(v) # fomo 加速度
        dev = MemeIndicators.pump_deviation(c)  # 价格偏离均线程度
        log_vol = torch.log1p(v) # torch.log1p(v) = log(1 + v),避免 v=0 时 log(0) 问题,同时压缩大成交量。
        
        # Advanced factors
        vol_cluster = MemeIndicators.volatility_clustering(c) # 波动率聚集(局部波动率)
        momentum_rev = MemeIndicators.momentum_reversal(c) # 动量反转(0 或 1)
        rel_strength = MemeIndicators.relative_strength(c, h, l) # 相对强弱(RSI 变体,-1～1)
        
        # High-low range 日内振幅占收盘价比例
        hl_range = (h - l) / (c + 1e-9)
        
        # Close position in range 收盘价在日内区间的位置(0～1)
        close_pos = (c - l) / (h - l + 1e-9)
            # close_pos 是经典“K线位置”指标：接近 1 表示强势收涨,接近 0 表示弱势收跌。
        
        # Volume trend 成交量变化率
        vol_prev = torch.roll(v, 1, dims=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        
        # 堆叠所有特征 
        features = torch.stack([
            self.robust_norm(ret),
            liq_score,
            pressure,
            self.robust_norm(fomo),
            self.robust_norm(dev),
            self.robust_norm(log_vol),
            self.robust_norm(vol_cluster),
            momentum_rev,
            self.robust_norm(rel_strength),
            self.robust_norm(hl_range),
            close_pos,
            self.robust_norm(vol_trend)
        ], dim=1)
        # torch.stack(..., dim=1) 将 12 个 [B, T] 张量堆叠成 [B, 12, T],
        # 这是时间序列模型(如 Transformer、CNN)的标准输入格式。
        
        return features


class FeatureEngineer:
    """
        从原始市场数据(如 K 线、流动性等)中高效提取一组标准化的 6 维技术特征(factors),
        专为高波动性加密资产(如 meme 币)设计。
        其核心目标是将原始行情数据转换为数值稳定、量纲统一、抗异常值的特征张量,可直接输入机器学习模型(如预测网络、强化学习策略等)。
    """
    INPUT_DIM = 6

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9)) #  对数收益率(Log Return)
        liq_score = MemeIndicators.liquidity_health(liq, fdv) # 流动性健康度
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l) # 买卖压力(K线实体强度)
        fomo = MemeIndicators.fomo_acceleration(v) # FOMO 情绪加速度
        dev = MemeIndicators.pump_deviation(c) # 价格偏离度(是否暴涨？)
        log_vol = torch.log1p(v) # 对数成交量
        
        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack([
            robust_norm(ret),
            liq_score,
            pressure,
            robust_norm(fomo),
            robust_norm(dev),
            robust_norm(log_vol)
        ], dim=1)
        
        return features
# data_loader.py
class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT address FROM tokens 
        LIMIT {limit_tokens} 
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")

# alphagpt.py
class NewtonSchulzLowRankDecay:
    """
    Low-Rank Decay (LoRD) using Newton-Schulz iteration.
    
    A more efficient regularization method that targets low-rank structure
    in attention and key parameters. Uses Newton-Schulz iteration to compute
    the minimum singular vectors without explicit SVD.
    
    Args:
        named_parameters: Model's named parameters
        decay_rate: Strength of low-rank decay
        num_iterations: Number of Newton-Schulz iterations (default: 5)
        target_keywords: If specified, only decay parameters matching these keywords

    名为 AlphaGPT 的神经网络模型,专为自动生成交易策略公式(如量化因子) 而设计。
        它结合了多种先进架构技术(如 Looped Transformer、QK-Norm、SwiGLU、MTP Head 等),
        并引入了低秩正则化(LoRD) 和稳定秩监控机制,
        以提升模型在小样本、高噪声金融数据上的泛化能力
    """
    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self):
        """Apply Newton-Schulz low-rank decay to attention parameters."""
        for name, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            # Transpose if needed for efficiency
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            # Normalize by spectral norm
            norm = X.norm() + 1e-8
            X = X / norm
            
            # Initialize Y for Newton-Schulz iteration
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
            # This converges to the orthogonal matrix with same singular vectors
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            # Apply low-rank decay
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    """Monitor the effective rank (stable rank) of model parameters."""
    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []
    
    @torch.no_grad()
    def compute(self):
        """Compute average stable rank of target parameters."""
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for Attention"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))
    
    def forward(self, q, k):
        # Normalize Q and K independently
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    """Swish GLU activation function"""
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)
    
    def forward(self, x):
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)  # Swish activation
        return self.fc(x)


class MTPHead(nn.Module):
    """Multi-Task Pooling Head for multi-objective learning"""
    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
        ])
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks)
        )
    
    def forward(self, x):
        # Route to appropriate task heads
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        
        # Compute all task outputs
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)  # [B, num_tasks, vocab_size]
        
        # Weighted combination
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
        return weighted, task_probs


class LoopedTransformerLayer(nn.Module):
    """Looped Transformer Layer - recurrent processing within a layer"""
    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead
        
        # QK-Norm attention
        self.qk_norm = QKNorm(d_model // nhead)
        
        # Standard attention components
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        # RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU FFN instead of standard FFN
        self.ffn = SwiGLU(d_model, dim_feedforward)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, is_causal=False):
        # Looped processing - recurrent refinement
        for _ in range(self.num_loops):
            # Self-attention with residual
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)
            
            # FFN with residual
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)
        
        return x


class LoopedTransformer(nn.Module):
    """Looped Transformer Encoder with multiple loop iterations"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        self.features_list = ['RET', 'VOL', 'V_CHG', 'PV', 'TREND']
        self.ops_list = [cfg[0] for cfg in OPS_CONFIG]
        
        self.vocab = self.features_list + self.ops_list
        self.vocab_size = len(self.vocab)
        
        # Embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ModelConfig.MAX_FORMULA_LEN + 1, self.d_model))
        
        # Enhanced Transformer with Looped Transformer
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            num_loops=3,
            dropout=0.1
        )
        
        # RMSNorm instead of LayerNorm
        self.ln_f = RMSNorm(self.d_model)
        
        # MTPHead for multi-task output
        self.mtp_head = MTPHead(self.d_model, self.vocab_size, num_tasks=3)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        # Process through looped transformer
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        last_emb = x[:, -1, :]
        
        # Multi-task pooling head for logits
        logits, task_probs = self.mtp_head(last_emb)
        value = self.head_critic(last_emb)
        
        return logits, value, task_probs
# vm.py

class StackVM:
    def __init__(self):
        """
        feat_offset = 6
            初始化映射表
            token 0~5 : 输入特征
            token 6+ 操作符
        """
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        # 将操作数token映射到函数
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        # 记录每个操作符需要几个参数

    def execute(self, formula_tokens, feat_tensor): # 执行引擎
        """            
            输入:
                formula_tokens: List[int] , 表示公式的token 序列(后缀表达式/逆波兰表达式)
                feat_tensor:  Tensor, shape[B,6,T], 由Feature Engineer.compute_feature() 生成.
            输出:
                成功: Tensor , shape[B,T] (每一个代币的时间序列信号)
                失败: None(语法错误,栈不匹配,NaN等)
            关键点：使用后缀表达式(RPN),天然适合栈机执行,无需括号。
        """
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None
# backtest.py
class MemeBacktest:
    """
        用于对 meme 币(高波动性加密货币)交易策略进行回测评估。
        它接收模型生成的信号(factors)、原始市场数据(raw_data)和目标收益率(target_ret),
        模拟真实交易环境(包括流动性限制、滑点、手续费等),
        并输出一个综合适应度分数(fitness score),常用于进化算法或强化学习中的策略评估.
    """
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060

    def evaluate(self, factors, raw_data, target_ret):
        """
            参数        类型            shape           解释
            factors:    torch.Tensor    [B,T]           模型输出的原始信号
            raw_data:   dict            --              包含'liquidity'等原始数据
            target_ret: torch.Tensor    [B,T]           目标持有收益率
        """
        # 生成交易信息
        liquidity = raw_data['liquidity']   # 流动性指标
        signal = torch.sigmoid(factors)     # 将信号压缩到(0,1)
        is_safe = (liquidity > self.min_liq).float()    # 流动性达标? 1= 安全,0=危险
        position = (signal > 0.85).float() * is_safe    #仅当信号强的企鹅安全的时候持仓
        # 计算交易成本(滑点+手续费)
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05) #最大滑点 5%
        total_slippage_one_way = self.base_fee + impact_slippage
        # 计算换手率和交易成本
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0 # 第一时间步无前序持仓
        turnover = torch.abs(position - prev_pos)   # 0->1 或 1->0 表示交易
        tx_cost = turnover * total_slippage_one_way
        # 计算盈亏
        gross_pnl = position * target_ret # 持仓期间毛收益
        net_pnl = gross_pnl - tx_cost       # 扣除交易成本后的净收益
        # 构建综合评分(fitness score)
        cum_ret = net_pnl.sum(dim=1)    # 每个代币的累计净收益
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)    # 单期亏损>5% 的次数
        score = cum_ret - (big_drawdowns * 2.0) #惩罚大回撤
        # 过滤低活跃度策略
        activity = position.sum(dim=1) # 总交易次数
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        # 最终适应度
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()        
# engine.py
class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        # ipdb.set_trace()
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }
    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        # ipdb.set_trace()
        
        for step in pbar:
            # ipdb.set_trace()
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                res = self.vm.execute(formula, self.loader.feat_tensor)
                
                if res is None:
                    rewards[i] = -5.0
                    continue
                
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                rewards[i] = score
                
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            
            loss = loss.mean()
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")

if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
