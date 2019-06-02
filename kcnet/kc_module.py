import torch
import torch.nn as nn

from .kc_functions import KernelCorrFunc, GraphMaxPoolingFunc

class KernelCorrelation(nn.Module):
    def __init__(self, num_k, num_kpts, dim, sigma, init_bound):
        super(KernelCorrelation, self).__init__()
        self.num_kernels = num_k
        self.num_kernel_pts = num_kpts
        self.kernel_dim = dim
        self.sigma = sigma
        self.init_kernel = init_bound

        self.weight = nn.Parameter(torch.Tensor(num_k, num_kpts, dim)) # LxMxD
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-self.init_kernel, self.init_kernel)


    def forward(self, input, indptr, indices):
        return KernelCorrFunc.apply(input, indptr, indices, self.weight, self.sigma)

    def extra_repr(self):
        return 'num_kernels={}, num_pts={}, num_dim={}'.format(
            self.num_kernels, self.num_kernel_pts, self.kernel_dim)

class GraphMaxPooling(nn.Module):
    def __init__(self):
        super(GraphMaxPooling, self).__init__()

    def forward(self, input, indptr, indices):
        return GraphMaxPoolingFunc.apply(input, indptr, indices)

class KCNet(nn.Module):
    def __init__(self, num_k, num_kpts, input_dim, sigma, init_bound, class_dim):
        super(KCNet, self).__init__()
        # self.kc = KernelCorrelation(num_k, num_kpts, input_dim, sigma, init_bound)
        # self.mlp1 = nn.Linear(num_k+input_dim, 64, bias=False)

        self.mlp1 = nn.Linear(input_dim, 64, bias=False)
        nn.init.xavier_uniform_(self.mlp1.weight)

        self.mlp2 = nn.Linear(64, 64, bias=False)
        nn.init.xavier_uniform_(self.mlp2.weight)

        self.gmp = GraphMaxPooling()

        self.mlp3 = nn.Linear(64, 64, bias=False)
        nn.init.xavier_uniform_(self.mlp3.weight)

        self.mlp4 = nn.Linear(64, 128, bias=False)
        nn.init.xavier_uniform_(self.mlp4.weight)

        self.mlp5 = nn.Linear(192, 1024, bias=False)
        nn.init.xavier_uniform_(self.mlp5.weight)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(512, 256, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(256, class_dim, bias=False)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.relu = nn.ReLU()

    def forward(self, x, indptr, indices):
        # kc_output = self.kc(x, indptr, indices)
        # x_concat = torch.cat([x, kc_output], dim=1) # Nx(L+3)
        # x_concat = x_concat.unsqueeze(0)  # 1xNx(L+3)

        x_concat = x.unsqueeze(0) # [1,N,3]

        x = self.mlp1(x_concat)
        x = self.relu(x)

        x = self.relu(self.mlp2(x))

        x = x.view(x.shape[1], -1)
        x_gm = self.gmp(x, indptr, indices) # Nx64
        x_gm = x_gm.unsqueeze(0) # 1xNx64
        x = x.unsqueeze(0) # 1xNx64

        x = self.relu(self.mlp3(x))

        x = self.relu(self.mlp4(x))

        x_concat2 = torch.cat([x_gm, x], dim=2) # 1xNx192

        x = self.relu(self.mlp5(x_concat2)) # BxNx1024

        x = torch.max(x, 1, keepdim=True)[0] # 1x1x1024

        x = x.view(-1,1024) # 1x1024

        x = self.relu(self.fc1(x))

        x = self.dropout1(x)

        x = self.relu(self.fc2(x))

        x = self.dropout2(x)

        x = self.fc3(x)  # 1xK

        return x
