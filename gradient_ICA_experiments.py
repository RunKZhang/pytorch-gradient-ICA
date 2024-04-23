import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from models.grica_model import grica_Encoder, MINE_Estimator, grica_model
import torch
from models.grica_model import alt_power_whitening
# from torchviz import make_dot

def generate_X():
    # Set a seed for the random number generator for reproducibility
    np.random.seed(23)

    # Number of samples
    ns = np.linspace(0, 8, 2000)

    # Source matrix
    S = np.array([np.sin(ns * 2), 
                  signal.sawtooth(ns * 3),
                  np.sign(np.sin(2 * np.pi *  ns))])
    S += 0.02*np.random.normal(size=S.shape)
    # S_raw = RawArray(S.T, info)
    
    # visulize S
    colors = ['red', 'steelblue', "orange"]
    # plt.figure(figsize=(8, 8))
    for sig, color in zip(S, colors):
        plt.plot(sig, color=color)
    plt.show()

    # print(S.shape)

    # mixing matrix
    A = np.array([[1, 1, 1],[0.5, 2, 1],[1.5, 1, 2]])
    X = np.dot(A, S)

    # visulize X
    colors = ['red', 'steelblue', "orange"]
    # plt.figure(figsize=(8, 8))
    for sig, color in zip(X, colors):
        plt.plot(sig, color=color)
    plt.show()
    return S, X

def test_whitening():
    # data to test whitening and Encoder
    num_samples = 1000
    input_dim = 5
    mean = np.zeros(input_dim)+0.1
    cov = np.random.randn(input_dim, input_dim)
    cov = np.dot(cov, cov.transpose())
    data = np.random.multivariate_normal(mean, cov, num_samples)
    data_tensor = torch.tensor(data, dtype = torch.float32)

    whitening_layer = alt_power_whitening(5, 250)
    whitened_output, W, mean, C = whitening_layer(data_tensor)
    print(f'Mean of data:{whitened_output.mean(0)}')
    cov_whitened = torch.matmul(whitened_output.T, whitened_output) / whitened_output.shape[0]
    print(f'Cov of data: {cov_whitened}')


def grica_experiments(device):
    # Source in shape (3, 800), 
    Source, Observe = generate_X()
    
    S_input = torch.tensor(Source.T, dtype=torch.float32).to(device)
    # print(f'S:{Source.shape}')
    encoder = grica_Encoder(input_dim=3, output_dim=3)
    estimator = MINE_Estimator(input_dim=3, hidden_dim=64)
    train_container = grica_model(encoder, estimator, device)
    # model(Source.T)
    
    # encoded = encoder(S_input)
    # print(encoded.grad_fn)
    # for name, param in encoder.named_parameters():
    #     print(name, param.grad_fn)
    # make_dot(encoded, params=dict(encoder.named_parameters()))
    # plt.show()

    # opt_a is in charge of encoder, opt_b is in charge of estimator
    # opt_a = torch.optim.Adam(train_container.encoder.parameters(), lr = 0.005)
    opt_a = torch.optim.Adam(encoder.parameters(), lr = 0.005)
    # opt_b = torch.optim.Adam(model.mine.parameters(), lr = 1e-3)
    opt_b = torch.optim.Adam(train_container.mine_estimator.parameters(), lr = 0.0001, weight_decay=0.001)
    
    # print before unmixing
    params = train_container.test(S_input)
    print(f'before unmixing matrix:\n {next(params).detach().cpu().numpy()}')


    # optimize and test
    train_container.optimize(400, S_input, opt_a, opt_b)

    # print after optimize
    params = train_container.test(S_input)
    
    print(f'after unmixing matrix:\n {next(params).detach().cpu().numpy()}')
    # print(f'vector:\n {next(params).detach().cpu().numpy()}')
    Z = train_container.encoder(S_input)
    colors = ['red', 'steelblue', "orange"]
    # plt.figure(figsize=(8, 8))
    for sig, color in zip(Z.detach().cpu().numpy().T, colors):
        plt.plot(sig, color=color)
    plt.show()
    
    
    # x = model.optimize(torch.Tensor(Source.T))
    # print(f'x shape: {x.shape}')
    # T = MINE_Estimator()
