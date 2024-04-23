import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import math
# from .utils import batch

EPS = 1e-6

# mi_lb = mutual information lower bound, t = T_theta, et = e^T_theta
# def mutual_information(joint, marginal, mine_net):
#     t = mine_net(joint)
#     et = torch.exp(mine_net(marginal))
#     mi_lb = torch.mean(t) - torch.log(torch.mean(et))
#     return mi_lb, t, et

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    
    # for keep it run, remove along 0 dim
    # t_exp = torch.exp(torch.logsumexp(x, (0,1)) - math.log(x.shape[0])).detach()

    # print(f't_exp in ema_loss: {t_exp.shape}')
    # print(f'torch.logsumexp(x, 0): {torch.logsumexp(x, 0).shape}')
    # print(f'math.log(x.shape[0]): {math.log(x.shape[0]).shape}')
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean

class Mine(nn.Module):
    def __init__(self, T, device='cuda:0'):
        super().__init__()
        self.running_mean = 0
        self.alpha = 0.01
        self.avg_test_mi = 0
        self.device = device

        self.T = T.to(device)
        
    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]


        # print(f'x shape:{x.shape}, z shape:{z.shape}, z_marg shape:{z_marg.shape}')
        t = self.T(x, z).mean()
        # print(f't shape:{t.shape}')
        # print(t)
        t_marg = self.T(x, z_marg)
        # print(f't_marg shape: {t_marg.shape}')
        
        # biased
        et_marg = torch.exp(t_marg)
        second_term = torch.log(torch.mean(et_marg))
        
        # unbiased
        # print(f'self.running_mean:{self.running_mean}')
        # second_term, self.running_mean = ema_loss(t_marg, self.running_mean, self.alpha)

        return -t + second_term
    
    def mi(self, x, z, z_marg=None):
        mi = -self.forward(x, z, z_marg)
        return mi
    
    def optimize(self, iters, dataloader, opt=None):
        for iter in range(1, iters + 1):
            print(f'iter_num:{iter}')
            mu_mi = 0
            # for x, y in batch(X, Y, batch_size):
            for _, (x,y) in enumerate(dataloader):
                opt.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                loss = self.forward(x, y)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()

    def test(self, dataloader):
        avg_mi = []
        with torch.no_grad():
            for _, (x,y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                mi = self.mi(x,y)
                avg_mi.append(mi)
        avg_mi = torch.Tensor(avg_mi)
        self.avg_test_mi = avg_mi.mean().detach().numpy()