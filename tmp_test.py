import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F



def entropy_torch(p, logp, q, logq):
    ans = p * (logp - logq)
    ans = torch.sum(ans, dim=-1)
    print('ans shape:', ans, ans.shape)
    return ans

def dist2_entropy_torch(a, b):
    

with torch.no_grad():
    a = torch.softmax(torch.randn((10)), dim=-1)
    b = torch.softmax(torch.randn((2, 10)), dim=-1)
    print(a, a.shape)
    print(b, b.shape)
    # a = a.unsqueeze(0).repeat((b.shape[0], 1))
    a = a.unsqueeze(0)
    print('padding a:', a, a.shape)
    loga = torch.log(a)
    logb = torch.log(b)
    ans_vec = entropy_torch(a, loga, b, logb) + entropy_torch(b, logb, a, loga)
    # ans_vec = torch.cosine_similarity(a, b, dim = -1)
    # ans_vec = F.kl_div(torch.log(a), b, reduction='mean') +F.kl_div(torch.log(b), a, reduction='none')
    # ans_vec = F.kl_div(a, b)
    # ans_vec = torch.dist(a, b, p=2)
    print(ans_vec, ans_vec.shape)



    # 定义两个矩阵
    x = torch.randn((4, 5))
    y = torch.randn((4, 5))
    # 因为要用y指导x,所以求x的对数概率，y的概率
    logp_x = F.log_softmax(x, dim=-1)
    p_y = F.softmax(y, dim=-1)

    print('logp_x:', logp_x, logp_x.shape)
    print('p_y:', p_y, p_y.shape)
    
    
    kl_sum = F.kl_div(logp_x, p_y, reduction='sum', dim = -1)
    kl_mean = F.kl_div(logp_x, p_y, reduction='mean', dim = -1)
    
    print(kl_sum, kl_mean)

    