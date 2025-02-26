import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import lightning.pytorch as pl

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

class MMAB(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln, dropout_ratio):
        super(MMAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_heads = num_heads
        self.dim_head = dim_V // num_heads

        self.W_q = nn.Linear(dim_Q, dim_V)
        self.W_k = nn.Linear(dim_K, dim_V)
        self.W_v = nn.Linear(dim_K, dim_V)

        self.dropout = nn.Dropout(dropout_ratio)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
            self.ln_q = nn.LayerNorm(dim_Q)
            self.ln_k = nn.LayerNorm(dim_K)

        self.W_o = nn.Linear(dim_V, dim_V, bias=True)

    def mask_attention(self, A, M):
        if M is not None:
            M = M.repeat(self.num_heads, 1, 1).to(device)
            A = A.masked_fill(M == 0, float(-1e30))
        return A
    
    def forward(self, Q, K, M):
        return self.compute_attention(Q, K, M)
    
    def compute_attention(self, Q, K, M):
        Q = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K = K if getattr(self, 'ln_k', None) is None else self.ln_k(K)

        Q = self.W_q(Q)
        K, V = self.W_k(K), self.W_v(K)

        Q_ = torch.cat(Q.split(self.dim_head, 2), 0)
        K_ = torch.cat(K.split(self.dim_head, 2), 0)
        V_ = torch.cat(V.split(self.dim_head, 2), 0)

        A = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_head)
        A = self.mask_attention(A, M)
        A = F.softmax(A, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.W_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
    
class MSAB(pl.LightningModule):
    def __init__(self, dim_in, dim_out, num_heads, ln, dropout_ratio):
        super(MSAB, self).__init__()
        self.mmab = MMAB(dim_in, dim_in, dim_out, num_heads, ln, dropout_ratio)

    def forward(self, X, M):
        return self.mmab(X, X, M)
    