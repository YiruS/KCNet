from __future__ import print_function

import numpy as np
import torch
from torch.autograd import gradcheck
from torch.autograd import Variable


class KernelCorrFunc(torch.autograd.Function):
    """
    Kernel correlation
    input:
        X: nx3, G: nxn
    output:
        Y: nx1
    """

    @staticmethod
    def forward(ctx, input, indptr, indices, weights, sigma):
        """
        :param input: Nx3
        :param indptr, indices: csr format for sparse matrix
        :param weights: LxMx3
        :param sigma: kernel width
        :return: NxL
        """
        # B = input.size()[0]  # number of batches
        N = input.size()[0]  # number of points
        D = input.size()[1]  # dimension of input
        L = weights.size()[0]  # number of output channels
        M = weights.size()[1]  # number of points in each kernel
        output = input.new(N,L)

        for s in torch.arange(0, N): # each point
            q = input[s,:] # (D,)
            ind_begin, ind_end = indptr[s], indptr[s+1]
            nbs = float(ind_end-ind_begin)
            for l in torch.arange(0,L):
                rl = input.new(1).fill_(0).squeeze(0)
                x_neigh = input[indices[ind_begin:ind_end].long(), :] # KxD
                x_diff = x_neigh - q.expand_as(x_neigh)
                for m in torch.arange(0,M):  # each kernel point
                    km = weights[l,m:m+1,:] # (1,D)
                    sum_over_dim = torch.sum(torch.mul(x_diff-km, x_diff-km), dim=1, keepdim=True)  # Kx1
                    sum_over_kpt = torch.sum(torch.exp(torch.div(-sum_over_dim, sigma)))  # scalar
                    rl += sum_over_kpt
                torch.div(rl, nbs, out=output[s,l])

        ctx.save_for_backward(input)
        ctx.indptr, ctx.indices = indptr, indices
        ctx.weights = weights
        ctx.sigma = sigma
        ctx.N, ctx.L, ctx.M, ctx.D = N, L, M, D

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        indptr, indices = ctx.indptr, ctx.indices
        weights = ctx.weights
        sigma = ctx.sigma
        grad_weights = weights.new(weights.size()).fill_(0) # LxMx3

        N, L, M, D = ctx.N, ctx.L, ctx.M, ctx.D

        for l in torch.arange(0,L):
            for m in torch.arange(0,M):
                dk_i = grad_weights[l,m,:] # (D,)
                k_i = weights[l,m,:] # (D,)
                for d in torch.arange(0,D):
                    for s in torch.arange(0,N):
                        dl_dcs = grad_output[s,l]
                        q = input[s,:] #(D,)
                        dcs_dki = input.new(1).fill_(0).squeeze(0)
                        ind_begin, ind_end = indptr[s], indptr[s+1]
                        nbs = torch.tensor([ind_end.float()-ind_begin.float()], dtype=torch.float32).squeeze(0)
                        for j in torch.arange(ind_begin,ind_end):
                            p = input[indices[j],:] # (D,)
                            dij = input.new(1).fill_(0).squeeze(0)
                            for dd in torch.arange(0,D):
                                vij = k_i[dd]+q[dd]-p[dd]
                                dij += vij*vij
                            fij = torch.exp(-dij/sigma)
                            vij = k_i[d]+q[d]-p[d]
                            dcs_dki += torch.mul(fij,vij)
                        dcs_dki *= (-2.0)/(sigma*nbs)
                        dk_i[d] += dl_dcs*dcs_dki

        return None, None, None, grad_weights, None


# ### gradient check for kernel correlation ###
# kc = KernelCorrFunc.apply
# adj = torch.tensor([[[0,1,0,1,0],[1,0,1,0,0],[0,1,0,0,1],[1,0,0,0,1],[0,0,1,1,0]],[[0,0,1,1,0],[0,0,0,1,1],[1,0,0,0,1],[1,1,0,0,0],[0,1,1,0,0]]], dtype=torch.double, requires_grad=False)
# input = (torch.randn(2,5,3, dtype=torch.double,requires_grad=False), adj, torch.randn(4,2,3, dtype=torch.double, requires_grad=True), torch.tensor(1, dtype=torch.double, requires_grad=False))
# test = gradcheck(kc, input, eps=1e-6, atol=1e-4)
# print(test)
# ### gradient check for kernel correlation ###


# kc = KernelCorrFunc.apply
# indptr = torch.tensor([0,  2,  4,  6,  8, 10], dtype=torch.int, requires_grad=False)
# indices = torch.tensor([1, 2, 0, 3, 0, 4, 1, 4, 2, 3], dtype=torch.int, requires_grad=False)
# input = (torch.randn(5,3, dtype=torch.double, requires_grad=False), indptr, indices, torch.randn(4,2,3, dtype=torch.double, requires_grad=True), torch.tensor(1, dtype=torch.double, requires_grad=False))
# test = gradcheck(kc, input, eps=1e-6, atol=1e-4)
# print("Gradient check of KC: ", test)

class GraphMaxPoolingFunc(torch.autograd.Function):
    """
    Graph Max Pooling
    input:
        X: nx3, G: nxn
    output:
        Y: nx3
    """

    @staticmethod
    def forward(ctx, input, indptr, indices):
        """
        :param input: BxNx3
        :param indptr, indices: csr format for sparse matrix
        :return: BxNx3
        """

        N = input.size()[0] # number of points
        D = input.size()[1] # dimension of input
        output = input.new(N, D)
        idx = torch.empty((N, D), dtype=torch.int, device=input.device)

        for s in torch.arange(0,N):
            ind_begin, ind_end = indptr[s], indptr[s+1]
            output[s,:] = input[s,:].clone()
            idx[s,:].fill_(int(s))
            neigh_add = torch.empty(ind_end - ind_begin + 1, dtype=torch.long, device=input.device)
            neigh_add[0:ind_end-ind_begin] = indices[ind_begin:ind_end]
            neigh_add[-1] = s
            mx_value, mx_idx = input[neigh_add,:].max(0)
            output[s,:] = mx_value
            idx[s,:] = neigh_add[mx_idx] # change from local index to global index


        ctx.save_for_backward(input)
        ctx.N, ctx.D = N, D
        ctx.indptr, ctx.indices, ctx.idx = indptr, indices, idx

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        N, D = ctx.N, ctx.D
        indptr, indices, idx = ctx.indptr, ctx.indices, ctx.idx
        grad_input = grad_output.new(N,D).fill_(0)

        zeros_vector = torch.zeros((1,D), dtype=grad_output.dtype, device=grad_output.device, requires_grad=False)
        for s in torch.arange(0,N):
            ind_begin, ind_end = indptr[s], indptr[s+1]
            grad_input[s, :] = torch.where(idx[s, :] == int(s), grad_output[s, :], zeros_vector)
            grad_input[s,:] += torch.where(idx[indices[ind_begin:ind_end].long(),:]==int(s), grad_output[indices[ind_begin:ind_end].long(),:], zeros_vector).sum(0)

        return grad_input, None, None


# ### gradient check for graph max pooling ###
# gm = GraphMaxPoolingFunc.apply
# indptr = torch.tensor([0,  2,  4,  6,  8, 10], dtype=torch.int, requires_grad=False)
# indices = torch.tensor([1, 2, 0, 3, 0, 4, 1, 4, 2, 3], dtype=torch.int, requires_grad=False)
# input = (torch.randn(5,3, dtype=torch.double,requires_grad=True), indptr, indices)
# #input = (torch.tensor([[[-0.49,-0.7,-0.54],[-0.86,-1.05,1.13],[0.43,1.89,0.87],[0.59,-0.2,0.26],[0.19,-0.45,-0.86]]], dtype=torch.double, requires_grad=True), adj)
# test = gradcheck(gm, input, eps=1e-6, atol=1e-4)
# print("Gradient check of Graph Max Pooling: ", test)
# ### gradient check for graph max pooling ###

