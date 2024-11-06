#Taken from https://github.com/kyg0910/CI-iVAE/blob/main/experiments/EMNIST_and_FashionMNIST/model.py

import torch
import torch.nn as nn

class Label_Prior(nn.Module):
    def __init__(self, dim_z, dim_u, hidden_nodes, leaky_relu_slope=0.2):
        super(Label_Prior, self).__init__()
        self.dim_z, self.dim_u, self.hidden_nodes = dim_z, dim_u, hidden_nodes
        
        # input dimension is dim_u
        self.main = nn.Sequential(
                        nn.Linear(self.dim_u, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
                        nn.Linear(self.hidden_nodes, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
                        )
        
        # input dimension is 20
        self.mean_net = nn.Linear(self.hidden_nodes, self.dim_z)
        self.log_var_net = nn.Linear(self.hidden_nodes, self.dim_z)
        
    def forward(self, u_input):
        h = self.main(u_input)
        mean, log_var = self.mean_net(h), self.log_var_net(h)
        return mean, log_var
    
class Label_Decoder(nn.Module):
    def __init__(self, dim_u, dim_z, hidden_nodes, leaky_relu_slope=0.2):
        super(Label_Decoder, self).__init__()
        self.dim_u, self.dim_z, self.hidden_nodes = dim_u, dim_z, hidden_nodes
        
        # input dimension is dim_u
        self.main = nn.Sequential(
                        nn.Linear(self.dim_z, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
                        nn.Linear(self.hidden_nodes, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
                        )
        
        # input dimension is 20
        self.out_net = nn.Linear(self.hidden_nodes, self.dim_u)
        
    def forward(self, u_input):
        h = self.main(u_input)
        return self.out_net(h)

def compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var):
    # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
    post_mean = (z_mean/(1+torch.exp(z_log_var-lam_log_var))) + (lam_mean/(1+torch.exp(lam_log_var-z_log_var)));
    post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var));
    
    return post_mean, post_log_var

def kl_criterion(mu1, log_var1, mu2, log_var2, reduce_mean=False):
    sigma1 = log_var1.mul(0.5).exp()
    sigma2 = log_var2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(log_var1) + (mu1 - mu2)**2)/(2*torch.exp(log_var2)) - 1/2
    return torch.mean(kld, dim=-1) if reduce_mean else torch.sum(kld, dim=-1)