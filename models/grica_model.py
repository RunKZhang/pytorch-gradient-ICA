import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from .MINE import Mine
from .networks import  ConcatLayer, CustomSequential
# 设置随机种子，以保证实验结果的可复现性
# torch.manual_seed(0)
# np.random.seed(0)

# def permute_y(y, N):
#     original_indices = torch.arange(N)
#     permuted_indices = original_indices[torch.randperm(N)]
#     y_permuted = y[permuted_indices]
#     return y_permuted

# def alt_power_whitening(self, input_tensor, output_dim, n_iterations=250, **kwargs):
#         R = torch.empty([output_dim,output_dim]).normal_(mean=0,std=1).cuda()
#         W = torch.zeros([output_dim,output_dim]).cuda()
#         input_tensor - input_tensor.mean(0)[None,:]
#         C = torch.matmul(input_tensor.T, input_tensor)/input_tensor.shape[0]
#         iter_count_tf = 0
#         condition = lambda it, C, W, R: it<output_dim
#         it = 0
#         while it<output_dim:
#             v, l = self.alt_matrix_power(C, R[:, it, None], n_iterations)
#             it+=1
#             C = C - l * torch.matmul(v, v.T)
#             W = W + 1 / torch.sqrt(l) * torch.matmul(v, v.T)
#         whitened_output = torch.matmul(input_tensor, W.T)
#         return whitened_output, W, input_tensor.mean(0), C

class alt_power_whitening(nn.Module):
    def __init__(self, output_dim, n_iterations):
        super().__init__()
        self.output_dim = output_dim
        self.n_iterations = n_iterations
        # self.R = nn.Parameter(torch.randn(output_dim, output_dim), requires_grad=False)
        # self.W = nn.Parameter(torch.zeros(output_dim, output_dim), requires_grad=False)
        self.R = torch.randn(output_dim, output_dim)
        self.W = torch.zeros(output_dim, output_dim)

    def single_power_step(self, A, x):
        x = torch.matmul(A, x)
        x = x/torch.norm(x)
        return x
    
    def alt_matrix_power(self, A, x, power):
        it = 0
        while it<power:
            it+=1
            x = self.single_power_step(A, x)
            # print(f'x inside: {x}')

        e = torch.norm(torch.matmul(A, x))
        return x, e
    
    def forward(self, x):
        input_mean = x.mean(dim=0)
        x = x - input_mean 
        C = x.t() @ x / x.size(0)
        
        it = 0
        while it<self.output_dim:
            v, l = self.alt_matrix_power(C, self.R[:, it, None], self.n_iterations)
            it+=1
            C = C - l * torch.matmul(v, v.T)
            self.W = self.W + 1 / torch.sqrt(l) * torch.matmul(v, v.T)
        whitened_output = torch.matmul(x, self.W.T)
        return whitened_output, self.W, x.mean(0), C

# class PowerWhitening(nn.Module):
#     def __init__(self, input_dim, output_dim, n_iterations):
#         super(PowerWhitening, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.n_iterations = n_iterations
#         self.R = nn.Parameter(torch.randn(output_dim, output_dim), requires_grad=False)
#         self.W = nn.Parameter(torch.zeros(output_dim, output_dim), requires_grad=False)

#     def forward(self, x):
#         input_mean = x.mean(dim=0)
#         x = x - input_mean
#         C = x.t() @ x / x.size(0)

#         for i in range(self.output_dim):
#             v, _ = self.power_iteration(C, self.R[:, i:i+1])
#             lambda_val = torch.norm(torch.matmul(C, v))
#             C -= lambda_val * (v @ v.t())
#             self.W.data += (1 / torch.sqrt(lambda_val)) * (v @ v.t())
#         whitened_output = x @ self.W.t()
#         return whitened_output

#     def power_iteration(self, A, x):
#         for _ in range(self.n_iterations):
#             x = torch.matmul(A, x)
#             x /= torch.norm(x)
#         return x, torch.norm(torch.matmul(A, x))

# class ICAEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_splits):
#         super(ICAEncoder, self).__init__()
#         self.dense = nn.Linear(input_dim, input_dim)
#         self.whitening = PowerWhitening(input_dim, input_dim, 50)
#         self.estimators = nn.ModuleList([nn.Linear(2 * input_dim, hidden_dim) for _ in range(7)])
#         self.n_splits = n_splits

#     def forward(self, x):
#         x = self.dense(x)
#         x = self.whitening(x)
        
#         # split to three parts
#         y1,y2,y3 = torch.split(x, x.shape[1] // 3, dim=1)
        
#         # rearange
#         x1 = torch.cat([y2,y3],dim=1)
#         x2 = torch.cat([y1,y3],dim=1)
#         x3 = torch.cat([y1,y2],dim=1)

#         y_1 = permute_y(y1, x.shape[0])
#         y_2 = permute_y(y2, x.shape[0])
#         y_3 = permute_y(y3, x.shape[0])

#         # estimate network
#         h_join1 = torch.cat([y1, x1], dim=1)
#         h_marg1 = torch.cat([y_1, x1], dim=1)

#         h_join2 = torch.cat([y2, x2], dim=1)
#         h_marg2 = torch.cat([y_2, x2], dim=1)

#         h_join3 = torch.cat([y3, x3], dim=1)
#         h_marg3 = torch.cat([y_3, x3], dim=1)
        
#         loss1 = self.estimate(h_join1, h_marg1)
#         loss2 = self.estimate(h_join2, h_marg2)
#         loss3 = self.estimate(h_join3, h_marg3)

#         return -(loss1+loss2+loss3)
    
#         # splits = torch.chunk(x, self.n_splits, dim=1)
#         # results = []
#         # for i in range(self.n_splits):
#         #     x_rest = torch.cat(splits[:i] + splits[i+1:], dim=1)
#         #     h_join = torch.cat([splits[i], x_rest], dim=1)
#         #     h_marg = torch.cat([torch.rand_like(splits[i]), x_rest], dim=1)  # 模拟排列
#         #     results.append(self.estimate(h_join, h_marg))
#         # return sum(results)

#     def estimate(self, h_join, h_marg):
#         for i, estimator in enumerate(self.estimators):
#             if i < len(self.estimators) - 1:
#                 h_join = F.leaky_relu(estimator(h_join))
#                 h_marg = F.leaky_relu(estimator(h_marg))
#             else:
#                 h_join = estimator(h_join)
#                 h_marg = estimator(h_marg)
#         # loss = torch.mean(h_join) - torch.log(torch.mean(torch.exp(h_marg)))
#         joint_output = torch.mean(h_join)
#         marginal_output = torch.log(torch.mean(torch.exp(h_marg)))
#         return joint_output - marginal_output



class grica_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim = output_dim
        # self.whitening = alt_power_whitening(output_dim, n_iterations=250).cuda()
    
    def forward(self, x):
        x = self.fc(x)
        x, _, _, _ = self.alt_power_whitening(x, self.output_dim)
        
        return x
    
    def single_power_step(self, A, x):
        x = torch.matmul(A, x)
        x = x/torch.norm(x)
        return x
    
    def alt_matrix_power(self, A, x, power):
        it = 0
        while it<power:
            it+=1
            x = self.single_power_step(A, x)
            # print(f'x inside: {x}')

        e = torch.norm(torch.matmul(A, x))
        return x, e
    
    def alt_power_whitening(self, input_tensor, output_dim, n_iterations=250):
        R = torch.empty([output_dim, output_dim]).normal_(mean=0,std=1).cuda()
        W = torch.zeros([output_dim, output_dim]).cuda()
        input_tensor - input_tensor.mean(0)[None,:]
        C = torch.matmul(input_tensor.T, input_tensor)/input_tensor.shape[0]
        iter_count_tf = 0
        condition = lambda it, C, W, R: it<output_dim
        it = 0
        while it<output_dim:
            v, l = self.alt_matrix_power(C, R[:, it, None], n_iterations)
            it+=1
            C = C - l * torch.matmul(v, v.T)
            W = W + 1 / torch.sqrt(l) * torch.matmul(v, v.T)
        whitened_output = torch.matmul(input_tensor, W.T)
        return whitened_output, W, input_tensor.mean(0), C
    

class MINE_Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # x_dim = 2
        # z_dim = 1 

        self.layer = CustomSequential(ConcatLayer(), 
                                      nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, 1))
        
    
    def forward(self, x, z):
        return self.layer(x, z)

class grica_model():
    def __init__(self, encoder, estimator, device):
        super().__init__()
        self.encoder = encoder.to(device)
        # self.estimator = estimator

        self.mine_estimator = Mine(estimator, device)

        self.z_list = []
        self.x_list = []

        self.device = device


    def optimize(self, iters, train_data, opt_a, opt_b):
        # opt_a is used for encoder, opt_b is used for estiamtor
        for iter in range(1, iters + 1):
            print(f'iter: {iter}')

            # pass once
            # train_data = train_data.to(self.device)
            # print(f'train_data: {train_data.device}, encoder: {self.encoder.whitening.R.device}')

            # forward pass, unmixing matrix
            encoded = self.encoder(train_data)
            
            # estimator_loss = 0
            encoder_loss = 0.0
            estimator_loss = 0.0
            
            
            if iter%7 == 0:
                for col_idx in range(encoded.shape[1]):
                    # extracted_tensor = encoded[:, col_idx:col_idx+1].clone()
                    extracted_tensor = encoded[:, col_idx:col_idx+1]
                    # remaining_tensor = torch.cat([encoded[:, :col_idx].clone(), encoded[:, col_idx+1:].clone()], dim=1)
                    remaining_tensor = torch.cat([encoded[:, :col_idx], encoded[:, col_idx+1:]], dim=1)
                    # estimator_loss = estimator_loss + self.mine_estimator(extracted_tensor.detach(), remaining_tensor.detach())
                    estimator_loss = estimator_loss + self.mine_estimator(extracted_tensor, remaining_tensor)
                    # estimator_loss = estimator_loss + self.mine_estimator(encoded[:, col_idx:col_idx+1].detach(), 
                    #                                                       torch.cat([encoded[:, :col_idx], encoded[:, col_idx+1:]], dim=1).detach())

                # encoder_loss = estimator_loss.clone()
                encoder_loss = - estimator_loss

                opt_a.zero_grad()
                encoder_loss.backward()
                opt_a.step()

            else:
                encoded_detach = encoded.detach()
                # make z_i, z_(-i), pass N times
                for col_idx in range(encoded.shape[1]):
                    # extracted_tensor = encoded_detach[:, col_idx:col_idx+1].clone()
                    extracted_tensor = encoded_detach[:, col_idx:col_idx+1]
                    # remaining_tensor = torch.cat([encoded_detach[:, :col_idx].clone(), encoded_detach[:, col_idx+1:].clone()], dim=1)
                    remaining_tensor = torch.cat([encoded_detach[:, :col_idx], encoded_detach[:, col_idx+1:]], dim=1)
                    estimator_loss = estimator_loss + self.mine_estimator(extracted_tensor, remaining_tensor)

                    # estimator_loss = estimator_loss + self.mine_estimator(encoded_detach[:, col_idx:col_idx+1],
                    #                                                       torch.cat([encoded_detach[:, :col_idx], encoded_detach[:, col_idx+1:]], dim=1))

                opt_b.zero_grad()
                estimator_loss.backward()
                opt_b.step()
                
            

            print(f'iter:{iter}, estimator_loss:{estimator_loss}, encoder_loss:{encoder_loss}')
        # return x
            
    def test(self, data):
        
        # data = data.to(self.device)
        # print(next(self.encoder.parameters()).device)
        # print(data.device)
        # encoded = self.encoder(data)
        # # print(encoded.shape)
        # colors = ['red', 'steelblue', "orange"]
        # # plt.figure(figsize=(8, 8))
        # for sig, color in zip(encoded.detach().cpu().numpy().T, colors):
        #     plt.plot(sig, color=color)
        # plt.show()
        # pass

        return self.encoder.parameters()
