import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_add_pool
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from math import sqrt
from utils import get_attn_pad_mask, create_ffn


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, modalities):
        """ modalities: [batch_size, num_modalities, dim] """
        batch_size = modalities.size(0)

        Q = self.q_proj(modalities).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(modalities).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(modalities).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attended = torch.matmul(attn_weights, V).transpose(1, 2)
        attended = attended.contiguous().view(batch_size, -1, self.dim)

        return self.out_proj(attended)


class EnhancedFusion(nn.Module):
    def __init__(self, dim=256, num_modalities=4):
        super().__init__()
        self.dim = dim
        self.attentions = nn.ModuleList([
            CrossModalAttention(dim) for _ in range(2)
        ])
        self.res_conn = nn.Linear(dim * num_modalities, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, modalities):
        residual = modalities.flatten(1)
        for attn in self.attentions:
            modalities = attn(modalities) + modalities
        fused = modalities.mean(dim=1)
        return self.norm(fused + self.res_conn(residual))


class Embed(nn.Module):
    def __init__(self, attn_head=4, output_dim=128, d_k=64, d_v=64, attn_layers=4, dropout=0.1, disw=1.5,
                 device='cuda:0'):
        super(Embed, self).__init__()
        self.device = device
        self.n_heads = attn_head
        self.relu = nn.ReLU()
        self.disw = disw
        self.layer_num = attn_layers
        self.gnns = nn.ModuleList([
            GraphConv(40, output_dim) if i == 0 else GraphConv(output_dim, output_dim)
            for i in range(attn_layers)])
        self.nms = nn.ModuleList([nn.LayerNorm(output_dim) for _ in range(attn_layers)])
        self.dps = nn.ModuleList([nn.Dropout(dropout) for _ in range(attn_layers)])
        self.tfs = nn.ModuleList([Encoder(output_dim, d_k, d_v, 1, attn_head, dropout) for _ in range(attn_layers)])
        self.proj = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch, leng, adj, dis):
        x = self.gnns[0](x, edge_index, edge_weight=edge_attr)
        x = self.dps[0](self.nms[0](x))
        x = self.relu(x)

        x_batch, mask = to_dense_batch(x, batch)

        batch_size, max_len, output_dim = x_batch.size()
        matrix_pad = torch.zeros((batch_size, max_len, max_len))
        for i, l in enumerate(leng):
            adj_ = torch.FloatTensor(adj[i]); dis_ = torch.FloatTensor(dis[i])
            adj_ = torch.where(adj_==1.2, torch.full_like(adj_, 1.1), adj_)
            dis_ = 1 / torch.pow(self.disw, (dis_ - 1))
            dis_ = torch.where(dis_ == self.disw, torch.zeros_like(dis_), dis_)
            matrix = torch.where(adj_ == 0, dis_, adj_)
            matrix_pad[i, :int(l[0]), :int(l[0])] = matrix
        matrix_pad = matrix_pad.unsqueeze(1).repeat(1, self.n_heads, 1, 1).to(self.device)

        x_batch = self.tfs[0](x_batch, mask, matrix_pad)
        for i in range(1, self.layer_num):
            x = torch.masked_select(x_batch, mask.unsqueeze(-1))
            x = x.reshape(-1, output_dim)
            x = self.gnns[i](x, edge_index, edge_weight=edge_attr)
            x = self.dps[i](self.nms[i](x))

            x = self.relu(x)
            x_batch, mask = to_dense_batch(x, batch)
            x_batch = self.tfs[i](x_batch, mask, matrix_pad)

        x_nodes = x_batch[mask]
        x_nodes = self.proj(x_nodes)
        return x_nodes


class GFE(nn.Module):
    def __init__(
            self,
            task='reg',
            tasks=1,
            attn_head=4,
            output_dim=128,
            d_k=64,
            d_v=64,
            attn_layers=4,
            D=16,
            dropout=0.1,
            disw=1.5,
            device='cuda:0',
            calc_feat_dim=12

    ):
        super(GFE, self).__init__()
        self.device = device
        self.emb = Embed(attn_head, output_dim, d_k, d_v, attn_layers, dropout, disw, device)
        self.fp_modle = FpModel()
        self.fused_dim = output_dim
        self.th = nn.Tanh()
        self.sm = nn.Softmax(-1)
        self.elec_fc = nn.Linear(3, output_dim)
        self.fp_proj = nn.Linear(128, output_dim)
        self.fusion = EnhancedFusion(dim=output_dim, num_modalities=4)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 4, 4),
            nn.Softmax(dim=1))
        self.pred_head = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, tasks)
        )
        self.calc_encoder = nn.Sequential(
            nn.Linear(12, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(
            self.device)
        leng, adj, dis = data.leng, data.adj, data.dis
        batch = data.batch.to(self.device)
        x_nodes = self.emb(x, edge_index, edge_attr, batch, leng, adj, dis)

        graph_feat = global_add_pool(x_nodes, batch)  # [batch_size, output_dim]

        fp1, fp2, fp3, fp4 = data.fp1.to(self.device), data.fp2.to(self.device), data.fp3.to(self.device), data.fp4.to(self.device)
        fp = self.fp_modle(fp1, fp2, fp3, fp4)
        fp = self.fp_proj(fp)

        electronic_feat = data.electronic_desc.to(self.device)  # [batch_size, 3]
        electronic_feat = self.elec_fc(electronic_feat)  # [batch_size, output_dim]
        qm9_feat = data.qm9_features.to(self.device)  # [batch_size, 12]
        qm9_feat = self.calc_encoder(qm9_feat)  # [batch_size, output_dim]

        modalities = torch.stack([graph_feat, fp, electronic_feat, qm9_feat], dim=1)

        fused_feat = self.fusion(modalities)  # [batch_size, 128]

        logits = self.pred_head(fused_feat)

        return torch.sigmoid(logits)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dp = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask, matrix):
        scores_ = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(self.d_k)
        scores = scores_ * matrix
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.sm(scores)
        context = torch.matmul(self.dp(attn), V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.nm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dp = nn.Dropout(p=dropout)
        self.sdpa = ScaledDotProductAttention(d_k, dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask, matrix):
        batch_size = input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn = self.sdpa(Q, K, V, attn_mask, matrix)
        # self.attn_weights = attn.detach().cpu().numpy()

        context = self.sdpa(Q, K, V, attn_mask, matrix)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)

        return self.dp(self.nm(output))  # self.attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.nm = nn.LayerNorm(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask, matrix):
        residual = enc_inputs
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, matrix)
        return self.nm(enc_outputs + residual)


class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_layers, n_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, mask, matrix):
        enc_self_attn_mask = get_attn_pad_mask(mask)
        for layer in self.layers:
            enc_inputs = layer(enc_inputs, enc_self_attn_mask, matrix)
        return enc_inputs


class FpModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(FpModel, self).__init__()
        self.fp1 = nn.Sequential(
            nn.Linear(881, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fp3 = nn.Sequential(
            nn.Linear(780, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fp4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3, x4):
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)
        combined = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 128*5=640]
        fused = self.fusion(combined)
        return fused
