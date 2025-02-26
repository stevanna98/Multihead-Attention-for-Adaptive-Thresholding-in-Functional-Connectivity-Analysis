import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math
import sys

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, sparser_num_heads, 
                 dropout_ratio, target_sparsity, sparsity_lambda):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_K
        self.sparser_num_heads = sparser_num_heads

        self.factor = 0.5

        self.target_sparsity = target_sparsity
        self.sparsity_lambda = sparsity_lambda

        self.conv_q = nn.Sequential(
            nn.Conv2d(1, sparser_num_heads, kernel_size=3, padding=1),
            nn.BatchNorm2d(sparser_num_heads),
            nn.ReLU(),
            nn.Conv2d(sparser_num_heads, sparser_num_heads * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(sparser_num_heads * 2),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(1, sparser_num_heads, kernel_size=3, padding=1),
            nn.BatchNorm2d(sparser_num_heads),
            nn.ReLU(),
            nn.Conv2d(sparser_num_heads, sparser_num_heads * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(sparser_num_heads * 2),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(1, sparser_num_heads, kernel_size=3, padding=1),
            nn.BatchNorm2d(sparser_num_heads),
            nn.ReLU(),
            nn.Conv2d(sparser_num_heads, sparser_num_heads * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(sparser_num_heads * 2),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )

    def sparsity_regularization(self, mask):
        current_sparsity = torch.mean((mask == 0).float(), dim=(1, 2))
        sparsity_loss = F.mse_loss(current_sparsity, 
                                  torch.tensor(self.target_sparsity, device=mask.device), 
                                  reduction='mean')
        return sparsity_loss * self.sparsity_lambda

    def forward(self, Q, K):
        Q_ = self.conv_k(Q.unsqueeze(1))
        K_ = self.conv_k(K.unsqueeze(1))
        V_ = self.conv_v(K.unsqueeze(1))

        head_outputs = []
        for head in range(self.sparser_num_heads * 2):
            q = Q_[:, head, :, :]
            k = K_[:, head, :, :]
            v = V_[:, head, :, :]

            A = F.softmax(q.bmm(k.transpose(1, 2)) / math.sqrt(self.sparser_num_heads * 2), 2)
            out = A.bmm(v)
            head_outputs.append(out)

        mask = torch.stack(head_outputs, dim=1)
        mask = torch.mean(mask, dim=1)

        B = mask.shape[0]  
        mask_flat = mask.view(B, -1)  
        median, _ = torch.median(mask_flat, dim=1, keepdim=True)  
        mad, _ = torch.median(torch.abs(mask_flat - median), dim=1, keepdim=True) 
        median = median.view(B, 1, 1)
        mad = mad.view(B, 1, 1)

        thr = median + self.factor * mad

        mask = torch.where(mask > thr, 1, torch.zeros_like(mask))

        sparsity_loss = self.sparsity_regularization(mask)

        return mask, sparsity_loss
    
    