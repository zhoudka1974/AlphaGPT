import torch

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