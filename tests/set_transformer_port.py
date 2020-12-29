import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorflow as tf
from models.set_transformer import SetTransformerEncoder


class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, num_heads, ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln))

        self.out_projection = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        x = self.enc(x)
        x = self.out_projection(x)
        return x


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


if __name__ == "__main__":
    set_data = [[[1, 2, 3, 3], [4, 5, 6, 3], [7, 8, 9, 3]], [[10, 11, 12, 3], [13, 14, 15, 3], [16, 17, 18, 3]]]

    input_dim = 4
    transformer_dim = 32
    output_dim = 2
    num_heads = 4

    tf_set = tf.constant(set_data, dtype=tf.float32)
    tf_set_transformer = SetTransformerEncoder(transformer_dim, num_heads, 1, output_dim)
    tf_out = tf_set_transformer(tf_set)

    torch_set = torch.tensor(set_data, dtype=torch.float32)
    torch_set_transformer = SetTransformer(input_dim, output_dim, transformer_dim, num_heads)
    out = torch_set_transformer(torch_set)