import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=0.1):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.initial_noise_std = noise_std
        self.noise_std = noise_std
        self.layer = nn.Linear(input_dim, num_experts)
    
    def set_noise_std(self, noise_std):
        self.noise_std = noise_std
    
    def forward(self, x, return_all_weights=False):
        logits = self.layer(x)
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        raw_weights = F.softmax(logits, dim=1)
        
        if self.top_k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(raw_weights, self.top_k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
            gate_weights = torch.zeros_like(raw_weights)
            gate_weights.scatter_(1, top_k_indices, top_k_weights)
        else:
            gate_weights = raw_weights
        
        if return_all_weights:
            return gate_weights, raw_weights
        return gate_weights


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        batch_size = query.shape[0]
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
        
        output = self.out_ln(attn_output + query)
        return output


class Encoder(nn.Module):
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([0.5])))
        
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        feat_map = self.input_norm(feat_map)
        
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0,2,1)  
              
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved+h_map)* self.scale
            h_map = conved
        
        pool_map = self.max_pool(h_map).squeeze(-1)
        h_map = h_map.permute(0,2,1)
        h_map = self.ln(h_map)
        return h_map, pool_map


class CAMDTA(nn.Module):
    def __init__(self, hp, device):
        super(CAMDTA, self).__init__()

        self.com_dim = hp.com_dim
        self.mlp_dim = hp.mlp_dim
        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim                
        self.hidden_dim = 128
        self.num_heads = 8
        
        self.num_experts = hp.num_experts
        self.top_k = hp.top_k
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)

        self.dropout = nn.Dropout(0.1)
        
        self.drug_embed = Encoder(hp.drug_max_len, self.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, self.protvec_dim, device)

        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        
        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)
        
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())     
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1)
            ) for _ in range(self.num_experts)
        ])
        
        self.gating = GatingNetwork(1024, self.num_experts, top_k=self.top_k, noise_std=self.moe_noise_std)
        
        self.register_buffer('expert_usage_count', torch.zeros(self.num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))
        

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)
        
        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)
        
        drug_cross_pool = self.drug_attn_pool(new_drug_embed)
        prot_cross_pool = self.prot_attn_pool(new_prot_embed)
        
        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        h_post = self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1))
        
        h_combined = h_pre + h_post
        
        gate_weights, raw_weights = self.gating(h_combined, return_all_weights=True)
        
        if self.training:
            with torch.no_grad():
                selected = (gate_weights > 0).float().sum(dim=0)
                self.expert_usage_count += selected
                self.total_samples += gate_weights.size(0)
        
        expert_outputs = []
        for i in range(self.num_experts):
            expert_pred = self.experts[i](h_combined)
            expert_outputs.append(expert_pred)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        pred = torch.sum(gate_weights_expanded * expert_outputs, dim=1)
        
        if return_gate_info:
            return pred, {
                'gate_weights': gate_weights,
                'raw_weights': raw_weights,
                'expert_outputs': expert_outputs.squeeze(-1),
                'selected_experts': (gate_weights > 0).int()
            }
        
        return pred
    
    def compute_load_balance_loss(self, gate_weights):
        expert_usage = gate_weights.mean(dim=0)
        ideal = 1.0 / self.num_experts
        load_balance_loss = self.num_experts * torch.sum(expert_usage ** 2)
        return load_balance_loss
    
    def get_expert_usage_stats(self):
        if self.total_samples == 0:
            return None
        
        usage_rate = self.expert_usage_count / self.total_samples
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        
        stats = {
            'expert_usage_rate': usage_rate.cpu().numpy(),
            'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item(),
            'dominant_expert': torch.argmax(usage_rate).item(),
            'usage_std': usage_rate.std().item(),
        }
        return stats
    
    def reset_usage_stats(self):
        self.expert_usage_count.zero_()
        self.total_samples.zero_()
    
    def adaptive_update(self, epoch, total_epochs, moe_stats=None):
        progress = epoch / total_epochs
        adjustments = {}
        
        initial_noise = self.gating.initial_noise_std
        new_noise = initial_noise * (1 - 0.9 * progress)
        self.gating.set_noise_std(new_noise)
        adjustments['noise_std'] = new_noise
        
        if moe_stats is not None:
            usage_std = moe_stats['usage_std']
            target_std = 0.15
            if usage_std > target_std * 1.5:
                self.load_balance_weight = min(self.load_balance_weight * 1.2, 0.1)
                adjustments['lb_adjustment'] = 'increased'
            elif usage_std < target_std * 0.5 and progress > 0.3:
                self.load_balance_weight = max(self.load_balance_weight * 0.9, 0.001)
                adjustments['lb_adjustment'] = 'decreased'
            else:
                adjustments['lb_adjustment'] = 'stable'
            adjustments['load_balance_weight'] = self.load_balance_weight
        
        return adjustments
