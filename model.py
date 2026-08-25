"""
Nowoczesna architektura Decoder-Only Transformer dla modelu LLM w PyTorch.
Wspiera Rotary Position Embeddings (RoPE), RMSNorm, FlashAttention (scaled_dot_product_attention) oraz próbkowanie tekstu.
"""

import math
import inspect
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (stosowane m.in. w LLaMA)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def apply_rope(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Aplikuje Rotary Position Embeddings (RoPE) do tensorów Q lub K o kształcie (B, n_head, T, head_dim)."""
    B, n_head, T, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim musi być parzyste dla RoPE"
    
    # Wyznaczenie częstotliwości theta
    half_dim = head_dim // 2
    theta = 10000.0 ** (-torch.arange(0, half_dim, dtype=torch.float32, device=x.device) / half_dim)
    seq_idx = torch.arange(T, dtype=torch.float32, device=x.device)
    idx_theta = torch.einsum('i,j->ij', seq_idx, theta)  # (T, half_dim)
    
    sin = idx_theta.sin()[None, None, :, :]  # (1, 1, T, half_dim)
    cos = idx_theta.cos()[None, None, :, :]  # (1, 1, T, half_dim)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    
    # Rotacja
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    
    return torch.cat([rx1, rx2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # Jedna macierz projekcji dla Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Projekcja wyjściowa
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch, Sequence Length, Embedding Dim

        # Obliczenie Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, T)
            k = apply_rope(k, T)

        # PyTorch FlashAttention / Scaled Dot-Product Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-Forward Network (z wybraną aktywacją SwiGLU / GELU)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 4x n_embd rozmiar warstwy ukrytej
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd) if config.bias is False else nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if config.bias is False else nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd) if config.bias is False else nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (podzielenie wag pomiędzy embedding a głowicę wyjściową)
        self.transformer.wte.weight = self.lm_head.weight

        # Inicjalizacja wag
        self.apply(self._init_weights)

        # Specjalne skalowanie wag dla połączeń rezidualnych wg nanoGPT / GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True) -> int:
        """Zwraca całkowitą liczbę parametrów w modelu."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        """Dzieli parametry na z grupy z weight decay (macierze 2D+) i bez decay (1D - biasy i normy)."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Liczba parametrów z weight decay: {len(decay_params)} ({num_decay_params:,})")
        print(f"Liczba parametrów bez weight decay: {len(nodecay_params)} ({num_nodecay_params:,})")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Używanie fused AdamW: {use_fused}")

        return optimizer

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Nie można przekazać sekwencji o długości {t}, max to {self.config.block_size}"

        # Token embedding
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)

        if not self.config.use_rope:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos) # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # Bloki transformera
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Obliczenie straty Cross Entropy
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # W trybie inferencji wystarczy obliczyć logity dla ostatniego tokena
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Autoregresywne generowanie tokenów z obsługą temperature, top-k oraz top-p (nucleus sampling)."""
        for _ in range(max_new_tokens):
            # Przycięcie kontekstu jeśli przekracza block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Usuń tokeny o skumulowanym prawdopodobieństwie powyżej progu top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Przesuń w prawo, aby zachować pierwszy token powyżej progu
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = -float('Inf')

            # Próbkowanie z rozkładu prawdopodobieństwa
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Dołączenie wygenerowanego tokena do sekwencji
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    # Szybki test jednostkowy architektury
    cfg = ModelConfig.get_preset("micro")
    model = LLM(cfg)
    print(f"Utworzono model preset 'micro'. Liczba parametrów: {model.get_num_params():,}")
    
    test_input = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(test_input, targets=test_input)
    print(f"Forward pass udany! Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
