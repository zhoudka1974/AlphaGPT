import torch
import torch.nn as nn


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