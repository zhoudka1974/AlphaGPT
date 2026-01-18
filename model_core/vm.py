import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer

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