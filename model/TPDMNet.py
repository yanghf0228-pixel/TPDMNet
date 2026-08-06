import torch
import torch.nn as nn
from model.TCDMU import TCDMU
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

class TimeAwareLocalSequentialEncoder(nn.Module):
    """Time-Aware Local Sequential Encoder (TLSE)."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1, num_time_basis=16, use_local_mask=False, dropout=0.1):
        super(TimeAwareLocalSequentialEncoder, self).__init__()
        assert num_layers >= 1, 'At least one TCDMU layer in TLSE.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.tcdmu_cells = nn.ModuleList()
        self.tcdmu_cells.append(TCDMU(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_time_basis, use_local_mask, dropout))
        for _ in range(1, num_layers):
            self.tcdmu_cells.append(TCDMU(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim, num_time_basis, use_local_mask, dropout))

    def forward(self, x, node_embeddings, local_mask=None):
        # x: [B, T, N, C]
        # output: [B, N, D]
        # TLSE acts as the time-aware local sequential encoder.
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        for i in range(self.num_layers):
            state = self.tcdmu_cells[i].init_hidden_state(x.shape[0])
            inner_states = []
            for t in range(seq_length):
                state = self.tcdmu_cells[i](
                    current_inputs[:, t, :, :],
                    state,
                    [node_embeddings[0][:, t, :], node_embeddings[1]],
                    local_mask=local_mask
                )
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: [B, T, N, hidden_dim]
        h = current_inputs[:, -1, :, :]  # Take the last timestep
        return h

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.tcdmu_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class TimeAwarePrototypeMemoryRetrieval(nn.Module):
    """Time-Aware Prototype Memory Retrieval Module (TPMR)."""

    def __init__(self, dim, num_prototypes, retrieval_mode="global", temperature=0.5):
        super(TimeAwarePrototypeMemoryRetrieval, self).__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.retrieval_mode = retrieval_mode
        self.prototype_bank = nn.Parameter(torch.randn(num_prototypes, dim), requires_grad=True)  # [num_prototypes, dim]
        nn.init.xavier_uniform_(self.prototype_bank)
        self.ema_decay = 0.995
        self.temperature = float(temperature)

    def update(self, h):
        # h: [B, N, D]
        # TPMR updates global traffic prototypes from pooled time-aware queries.
        with torch.no_grad():
            B, N, D = h.shape
            g = h.mean(dim=1)  # [B, D]
            g = F.layer_norm(g, g.shape[1:])
            logits = g @ self.prototype_bank.T  # [B, M]
            _, top1_idx = torch.topk(logits, 1, dim=-1)  # [B, 1]
            assign = torch.zeros_like(logits)
            assign.scatter_(1, top1_idx, 1.0)
            proto_sum = assign.T @ g  # [M, D]
            proto_count = assign.sum(dim=0)  # [M]
            proto_mean = proto_sum / torch.clamp(proto_count.unsqueeze(-1), min=1e-6)  # [M, D]
            valid = (proto_count > 0).unsqueeze(-1)  # [M, 1]
            updated = self.ema_decay * self.prototype_bank + (1 - self.ema_decay) * proto_mean
            self.prototype_bank.copy_(torch.where(valid, updated, self.prototype_bank))

    def retrieve(self, query):
        # query: [B, N, D]
        B, N, D = query.shape
        p = F.normalize(self.prototype_bank, dim=-1)
        if self.retrieval_mode == "node":
            q = F.normalize(query, dim=-1)
            logits = torch.matmul(q, p.T)  # [B, N, M]
            attn = F.softmax(logits / self.temperature, dim=-1)
            retrieved = torch.matmul(attn, self.prototype_bank)  # [B, N, D]
            return retrieved, attn
        if self.retrieval_mode == "global":
            g = query.mean(dim=1)  # [B, D]
            q = F.normalize(g, dim=-1)
            logits = q @ p.T  # [B, M]
            attn = F.softmax(logits / self.temperature, dim=-1)  # [B, M]
            attn = attn / attn.sum(dim=-1, keepdim=True)
            m_global = attn @ self.prototype_bank  # [B, D]
            retrieved = m_global.unsqueeze(1).expand(B, N, D)
            attn_node = attn.unsqueeze(1).expand(B, N, self.num_prototypes)
            return retrieved, attn_node
        raise ValueError(f"Unknown memory retrieval_mode: {self.retrieval_mode}")

    def prototype_diversity_loss(self):
        proto = F.normalize(self.prototype_bank, dim=-1)  # [M, D]
        sim = torch.matmul(proto, proto.T)  # [M, M]
        I = torch.eye(sim.size(0), device=sim.device)
        loss = (sim - I).pow(2).mean()
        margin = 0.2
        penalty = F.relu(sim - margin)
        loss = loss + penalty.mean()
        return loss

    def prototype_usage_loss(self, attn):
        # attn: [B, N, M]
        usage = attn.mean(dim=(0, 1))  # [M]
        target = torch.ones_like(usage) / usage.shape[0]
        return ((usage - target) ** 2).mean()

class LocalGlobalFusion(nn.Module):
    """Gated local-global fusion between TLSE states and TPMR prototype priors."""

    def __init__(self, dim, gate_bias=-3.0, memory_scale=1.0):
        super(LocalGlobalFusion, self).__init__()
        self.gate_proj = nn.Linear(2 * dim, dim)
        nn.init.constant_(self.gate_proj.bias, gate_bias)
        self.residual_scale = 1.0
        self.memory_scale = float(memory_scale)

    def forward(self, h, m):
        fused = torch.cat([h, m], dim=-1)
        gate = self.residual_scale * torch.sigmoid(self.gate_proj(fused))  # [B, N, D]
        # Local-global fusion uses the retrieved prototype prior to modulate local states.
        h_tilde = h + self.memory_scale * gate * m
        return h_tilde, gate

class DynamicGraphLearning(nn.Module):
    """Dynamic graph learning stage of DGLP."""

    def __init__(self, dim):
        super(DynamicGraphLearning, self).__init__()
        self.dim = dim
        self.temperature = 0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, h, local_mask=None):
        joint = self.proj(h)  # [B, N, D]
        joint = F.normalize(joint, dim=-1)
        sim = torch.matmul(joint, joint.transpose(-1, -2))  # [B, N, N]
        # Dynamic graph learning uses prototype-enhanced node representations.
        if local_mask is not None:
            local_mask = local_mask.to(device=h.device)
            sim = sim.masked_fill(local_mask.unsqueeze(0) == 0, -1e9)
        A_dyn = F.softmax(sim / self.temperature, dim=-1)
        return A_dyn

class DynamicGraphPropagation(nn.Module):
    """Graph propagation stage of DGLP."""

    def __init__(self, dim, num_layers=1):
        super(DynamicGraphPropagation, self).__init__()
        self.layer = nn.Linear(dim, dim)

    def forward(self, x, A_dyn):
        z = self.layer(torch.matmul(A_dyn, x))
        z = F.relu(z)
        return x + z

class PredictionHead(nn.Module):
    def __init__(self, dim, output_dim, horizon):
        super(PredictionHead, self).__init__()
        self.horizon = horizon
        self.linear = nn.Linear(dim, horizon * output_dim)

    def forward(self, z):
        pred = self.linear(z)  # [B, N, horizon * output_dim]
        pred = pred.view(z.shape[0], z.shape[1], self.horizon, -1)
        return pred

class TPDMNet(nn.Module):
    """Temporal Prototype-guided Dynamic Memory Network."""

    def __init__(self, args):
        super(TPDMNet, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.time_dim = args.time_dim
        self.cheb_k = args.cheb_k
        self.dropout_p = float(getattr(args, "dropout", 0.1))
        self.memory_retrieval = str(getattr(args, "memory_retrieval", "global")).lower()
        self.num_prototypes = int(getattr(args, "num_prototypes", 64))
        self.memory_temperature = float(getattr(args, "memory_temperature", 0.5))
        self.memory_gate_bias = float(getattr(args, "memory_gate_bias", -3.0))
        self.memory_scale = float(getattr(args, "memory_scale", 1.0))

        # Local mask for TLSE/DGLP: prefer physical adjacency.
        # If unavailable, pass None rather than inventing a fake node-index topology.
        adj_mx = getattr(args, "adj_mx", None)
        local_mask = None
        adj_candidates = list(adj_mx) if isinstance(adj_mx, (list, tuple)) else [adj_mx]
        for adj in adj_candidates:
            if adj is None:
                continue
            try:
                if hasattr(adj, "todense"):
                    adj = adj.todense()
                mask = adj.detach().clone().float() if torch.is_tensor(adj) else torch.as_tensor(np.asarray(adj), dtype=torch.float32)
            except (TypeError, ValueError):
                continue
            if mask.dim() == 2 and mask.shape[0] == self.num_node and mask.shape[1] == self.num_node:
                local_mask = (mask > 0).float()
                local_mask.fill_diagonal_(1.0)
                break

        self.has_local_mask = local_mask is not None and not (self.num_node > 2 and torch.all(local_mask == 1))
        if self.has_local_mask:
            identity = torch.eye(self.num_node, dtype=local_mask.dtype, device=local_mask.device)
            local_mask = (((local_mask > 0).float() + identity) > 0).float()
            local_mask = (torch.matmul(local_mask, local_mask) > 0).float()
        else:
            local_mask = torch.ones(self.num_node, self.num_node)
        self.register_buffer("local_mask", local_mask)

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.randn(288, self.time_dim), requires_grad=True)
        self.D_i_W_emb = nn.Parameter(torch.randn(7, self.time_dim), requires_grad=True)

        self.tlse = TimeAwareLocalSequentialEncoder(self.num_node, self.input_dim, self.hidden_dim, self.cheb_k, self.embed_dim, self.time_dim, self.num_layers, use_local_mask=True, dropout=self.dropout_p)
        self.tpmr = TimeAwarePrototypeMemoryRetrieval(
            self.hidden_dim,
            self.num_prototypes,
            retrieval_mode=self.memory_retrieval,
            temperature=self.memory_temperature,
        )
        self.prototype_query_proj = nn.Linear(self.hidden_dim + self.time_dim, self.hidden_dim)
        self.local_global_fusion = LocalGlobalFusion(
            self.hidden_dim,
            gate_bias=self.memory_gate_bias,
            memory_scale=self.memory_scale,
        )
        self.dynamic_graph_learning = DynamicGraphLearning(self.hidden_dim)
        self.dynamic_graph_propagation = DynamicGraphPropagation(self.hidden_dim)
        self.prediction_head = PredictionHead(self.hidden_dim, self.output_dim, self.horizon)

    def _time_embedding(self, data):
        t_i_d_data = data[..., -2]
        d_i_w_data = data[..., -1]
        T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).clamp(0, 287).long().to(data.device)]
        D_i_W_emb = self.D_i_W_emb[d_i_w_data.clamp(0, 6).long().to(data.device)]
        return torch.mul(T_i_D_emb, D_i_W_emb), T_i_D_emb

    def forward(self, source, target=None, batches_seen=None, return_aux=False):
        # source: [B, T, N, D]
        node_embedding, T_i_D_emb = self._time_embedding(source)

        en_node_embeddings = [node_embedding, self.node_embeddings]

        source = source[..., :self.input_dim]
        local_mask = self.local_mask if self.has_local_mask else None

        # Keep the physical adjacency isolated to DGLP; TLSE uses the sequential local encoder.
        h = self.tlse(source, en_node_embeddings, local_mask=None)  # [B, N, D]

        # Use the same time-aware query space for memory update and retrieval.
        memory_query = torch.cat([h, T_i_D_emb[:, -1, :, :]], dim=-1)
        memory_query = self.prototype_query_proj(memory_query)

        # Update memory only in training
        if self.training:
            self.tpmr.update(memory_query)

        # Retrieve with time-of-day prior for global traffic prototypes.
        m, memory_attn = self.tpmr.retrieve(memory_query)  # m: [B, N, D], memory_attn: [B, N, num_prototypes]

        # Memory modulates the local encoder state; encoder remains the main representation.
        h_tilde, fusion_gate = self.local_global_fusion(h, m)  # h_tilde: [B, N, D], fusion_gate: [B, N, D]
        with torch.no_grad():
            h_norm = h.norm(dim=-1).mean()
            m_norm = m.norm(dim=-1).mean()
            self.debug_stats = {
                "h_norm": h_norm.detach(),
                "m_norm": m_norm.detach(),
                "m_over_h": (m_norm / (h_norm + 1e-8)).detach(),
                "gate_mean": fusion_gate.mean().detach(),
                "attn_max": memory_attn.mean(dim=(0, 1)).max().detach(),
            }

        # Dynamic Graph Learning and Propagation (DGLP)
        A_dyn = self.dynamic_graph_learning(h_tilde, local_mask=local_mask)  # [B, N, N]

        # Graph propagation acts on the memory-modulated encoder representation.
        z = self.dynamic_graph_propagation(h_tilde, A_dyn)  # [B, N, D]

        pred = self.prediction_head(z)  # [B, N, horizon, output_dim]
        pred = pred.permute(0, 2, 1, 3)  # [B, horizon, N, output_dim] to match trainer

        if return_aux:
            return pred, memory_attn, A_dyn, fusion_gate
        return pred

# Deprecated legacy code - do not use in new model
# class Parallel_decoder(nn.Module):
#     ...

# class Recurrent_decoder(nn.Module):
#     ...

# class TransformAttentionModel(torch.nn.Module):
#     ...


