import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math


class TCDMN(nn.Module):
    """Time-Conditioned Dynamic Message Network from the TPDMNet paper."""

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_time_basis=16, use_local_mask=False, dropout=0.1):
        super(TCDMN, self).__init__()
        self.cheb_k = cheb_k
        self.use_local_mask = use_local_mask
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))

        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.fc1=nn.Sequential(
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             ('relu1', nn.ReLU()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             ('relu2', nn.ReLU()),
                             ('fc3', nn.Linear(self.middle_dim, self.time_dim))]))
        self.fc2=nn.Sequential(
                OrderedDict([('fc1', nn.Linear(self.time_dim, self.middle_dim)),
                             ('relu1', nn.ReLU()),
                             ('fc2', nn.Linear(self.middle_dim, self.hyperGNN_dim)),
                             ('relu2', nn.ReLU()),
                             ('fc3', nn.Linear(self.hyperGNN_dim, dim_in))]))

        self.time_basis = nn.Parameter(torch.randn(num_time_basis, self.time_dim), requires_grad=True)
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # self.fc3= nn.Linear(dim_in, dim_out, bias=True)
    def forward(self, x, node_embeddings, local_mask=None):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # TCDMN selects time-conditioned basis messages and applies node-adaptive transformations.
        assert x.dim() == 3, f"Expected x to be 3D, got {x.shape}"
        B, N, C = x.shape
        assert node_embeddings[1].shape[0] == N, \
            f"Node mismatch: x has {N}, embedding has {node_embeddings[1].shape[0]}"


        filter1 = self.fc1(x)                     # [B, N, time_dim]
        nodevec1 = filter1

        nodevec2 = torch.mul(
            node_embeddings[0].unsqueeze(-2),     # [B, N, 1, time_dim]
            self.time_basis                       # [K, time_dim]
        )                                         # -> [B, N, K, time_dim]

        cata = self.fc2(nodevec2)                # [B, N, K, dim_in]

        # 用 einsum 计算每个节点对 K 个 time basis 的相似度
        supports2 = torch.softmax(
            torch.einsum("bnd,bnkd->bnk", nodevec1, nodevec2),
            dim=-1
        )                                         # [B, N, K]

        # 按 K 个 basis 聚合
        if self.use_local_mask and local_mask is not None:
            local_mask = local_mask.to(device=x.device).float()
            if local_mask.shape == supports2.shape[-2:]:
                mask_weight = local_mask.unsqueeze(0)
            else:
                # supports2 is [B, N, K], while the physical mask is usually [N, N].
                # Use node-wise local degree as a compatible local prior when K != N.
                mask_weight = local_mask.sum(dim=-1, keepdim=True)
                mask_weight = mask_weight / torch.clamp(mask_weight.max(), min=1.0)
                mask_weight = mask_weight.reshape(1, N, 1)
            supports2 = supports2 * mask_weight
            supports2 = supports2 / (supports2.sum(dim=-1, keepdim=True) + 1e-8)

        x_g2 = torch.einsum("bnk,bnkc->bnc", supports2, cata)   # [B, N, dim_in]
        x_g2 = self.dropout(x_g2)

        x_g = torch.stack([x,x_g2],dim=1)


        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]


        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = self.dropout(x_gconv)
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        # x_gconv =self.fc3(x)

        return x_gconv * self.residual_scale

    @staticmethod
    def get_laplacian(graph,normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L
