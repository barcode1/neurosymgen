import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np


class NeuroSymGenLayer(nn.Module):
    """
    Core Neuro-Symbolic Layer with Hardware-Optimized Gated Fusion.
    ISCAI 2025: Supports torch.compile and TensorRT.
    """

    # def __init__(self, input_size: int, num_rules: int = 8, output_size: int = 128):
    #     super().__init__()
    #
    #     # Neural pathway
    #     self.neural_part = nn.Linear(input_size, output_size)
    #
    #     # Symbolic rules with learned logic matrix
    #     self.num_rules = num_rules
    #     self.rule_weights = nn.Parameter(torch.randn(num_rules))
    #     self.logic_matrix = nn.Parameter(torch.randn(num_rules, input_size))
    #     self.rule_mapper = nn.Linear(num_rules, output_size)
    #
    #     # ⭐ Gated Fusion Mechanism (ISCAI Innovation)
    #     self.fusion_gate = nn.Sequential(
    #         nn.Linear(output_size * 3, output_size),
    #         nn.Sigmoid()
    #     )
    #
    #     # ⭐ KG Integration with Attention
    #     self.kg_attention = nn.MultiheadAttention(
    #         embed_dim=output_size, num_heads=8, batch_first=True, dropout=0.1
    #     )
    #
    #     # ⭐ Self-explainability buffer
    #     self.register_buffer("activation_history", torch.zeros(10000, output_size))
    #     self.register_buffer("gate_history", torch.zeros(10000, 3))  # neural, symbolic, kg
    #     self.history_ptr = 0
    #
    #     self.output_size = output_size
    def __init__(self, input_size: int, num_rules: int = 8, output_size: int = 128,
                 num_heads: int = 8, final_output_size: int = None):
        """
        Args:
            input_size: Input dimension
            num_rules: Number of symbolic rules
            output_size: Internal representation size (must be divisible by num_heads)
            num_heads: Number of attention heads (default: 8)
            final_output_size: Final output size (e.g., 1 for binary classification)
        """
        super().__init__()

        # Neural pathway
        self.neural_part = nn.Linear(input_size, output_size)

        # Symbolic rules
        self.num_rules = num_rules
        self.rule_weights = nn.Parameter(torch.randn(num_rules))
        self.logic_matrix = nn.Parameter(torch.randn(num_rules, input_size))
        self.rule_mapper = nn.Linear(num_rules, output_size)

        # Gated Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_size * 3, output_size),
            nn.Sigmoid()
        )

        # KG Attention با num_heads قابل تنظیم
        if output_size % num_heads != 0:
            # محاسبه خودکار num_heads اگه قابل تقسیم نبود
            num_heads = max(1, output_size // 64)
            print(f"⚠️ Adjusting num_heads to {num_heads} for output_size={output_size}")

        self.kg_attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Final projection برای output size دلخواه
        if final_output_size is not None:
            self.output_projection = nn.Linear(output_size, final_output_size)
        else:
            self.output_projection = nn.Identity()

        # Buffers
        self.register_buffer("activation_history", torch.zeros(10000, output_size))
        self.register_buffer("gate_history", torch.zeros(10000, 3))
        self.history_ptr = 0

        self.output_size = output_size

    def forward(self, x: torch.Tensor, kg_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass
        Returns:
            output: [batch] یا [batch, final_output_size]
            info: dict
        """
        batch_size = x.size(0)

        # Neural pathway
        neural_out = self.neural_part(x)  # [batch, output_size]

        # Symbolic pathway
        rules_applied = torch.sigmoid(torch.matmul(x, self.logic_matrix.T))  # [batch, num_rules]
        rule_scores = rules_applied * self.rule_weights.unsqueeze(0)
        symbolic_out = self.rule_mapper(rule_scores)

        # KG pathway
        kg_out = torch.zeros_like(neural_out)
        if kg_data is not None:
            # اضافه کردن batch dimension اگه نداشت
            if kg_data.dim() == 2:
                kg_data = kg_data.unsqueeze(0).expand(batch_size, -1, -1)

            # KG Attention
            query = neural_out.unsqueeze(1)  # [batch, 1, output_size]
            kg_out, attn_weights = self.kg_attention(query, kg_data, kg_data)
            kg_out = kg_out.squeeze(1)

        # Gated Fusion
        gate_input = torch.cat([neural_out, symbolic_out, kg_out], dim=-1)
        gates = self.fusion_gate(gate_input)
        combined = gates * neural_out + (1 - gates) * symbolic_out + kg_out

        # Final projection
        combined = self.output_projection(combined)

        # Store history
        self._store_history(combined, gates, rules_applied)

        return torch.sigmoid(combined.squeeze(-1)), {  # squeeze برای [batch, 1]
            "rules": rules_applied,
            "gates": gates.mean(dim=0),
            "kg_attention": attn_weights if kg_data is not None else None
        }

    # def forward(self, x: torch.Tensor, kg_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    #     """
    #     Forward pass optimized for torch.compile.
    #
    #     Args:
    #         x: [batch, input_size]
    #         kg_data: [batch, num_nodes, output_size]
    #
    #     Returns:
    #         output: [batch, output_size]
    #         info: Dictionary with interpretability info
    #     """
    #     batch_size = x.size(0)
    #
    #     # Neural pathway
    #     neural_out = self.neural_part(x)
    #
    #     # Symbolic pathway
    #     # rules_applied = torch.sigmoid(torch.matmul(x, self.logic_matrix.T))  # [batch, num_rules]
    #     rules_applied = torch.sigmoid(x @ self.logic_matrix.T)
    #     rule_scores = rules_applied * self.rule_weights.unsqueeze(0)
    #     symbolic_out = self.rule_mapper(rule_scores)
    #
    #     # KG pathway
    #     kg_out = torch.zeros_like(neural_out)
    #     if kg_data is not None:
    #         # Reshape for attention
    #         query = neural_out.unsqueeze(1)  # [batch, 1, output_size]
    #         kg_out, attn_weights = self.kg_attention(query, kg_data, kg_data)
    #         kg_out = kg_out.squeeze(1)  # [batch, output_size]
    #
    #     # ⭐ Gated Fusion
    #     gate_input = torch.cat([neural_out, symbolic_out, kg_out], dim=-1)
    #     gates = self.fusion_gate(gate_input)  # [batch, output_size]
    #
    #     # ⭐ Weighted combination
    #     combined = gates * neural_out + (1 - gates) * symbolic_out + kg_out
    #
    #     # ⭐ Store for explainability
    #     self._store_history(combined, gates, rules_applied)
    #
    #     return torch.sigmoid(combined), {
    #         "rules": rules_applied,
    #         "gates": gates.mean(dim=0),  # Average gates for monitoring
    #         "kg_attention": attn_weights if kg_data is not None else None
    #     }

    def _store_history(self, activations: torch.Tensor, gates: torch.Tensor, rules: torch.Tensor):
        """Circular buffer for activation analysis"""
        n = activations.size(0)
        if self.history_ptr + n > self.activation_history.size(0):
            self.history_ptr = 0

        idx = self.history_ptr
        self.activation_history[idx:idx + n] = activations.detach()
        self.gate_history[idx:idx + n] = gates.detach().mean(dim=1, keepdim=True).expand(-1, 3)
        self.history_ptr += n

    def get_explainability_report(self) -> Dict[str, float]:
        """Generate interpretability metrics"""
        history = self.activation_history[:self.history_ptr]
        gates = self.gate_history[:self.history_ptr]

        return {
            "avg_neural_gate": gates[:, 0].mean().item(),
            "avg_symbolic_gate": gates[:, 1].mean().item(),
            "avg_kg_gate": gates[:, 2].mean().item(),
            "activation_sparsity": (history.abs() < 0.1).float().mean().item()
        }

    def reset_history(self):
        """Reset explainability buffers"""
        self.history_ptr = 0