"""
Konfiguracja modelu LLM oraz parametrów procesu treningowego.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 50304  # Domyślnie GPT-2 vocab 50257 zaokrąglone do najbliższej wielokrotności 64 dla szybszego matmul
    block_size: int = 1024   # Długość okna kontekstu (max sequence length)
    n_layer: int = 12        # Liczba warstw Transformera
    n_head: int = 12         # Liczba głowic uwagi
    n_embd: int = 768        # Wymiar ukryty (embedding dimension)
    dropout: float = 0.0     # Dropout (0.0 dla większych zbiorów, 0.1 do mniejszych)
    bias: bool = False       # Czy używać biasu w Linear i LayerNorm (LLaMA style: False)
    use_rope: bool = True    # Czy używać Rotary Position Embeddings (RoPE) zamiast learned absolute pos emb

    @classmethod
    def get_preset(cls, name: str) -> "ModelConfig":
        """Zwraca gotowy preset modelu."""
        presets = {
            # ~10M parametrów (super szybki do testów)
            "micro": cls(
                n_layer=4,
                n_head=4,
                n_embd=256,
                block_size=512,
            ),
            # ~45M parametrów
            "mini": cls(
                n_layer=8,
                n_head=8,
                n_embd=512,
                block_size=1024,
            ),
            # ~124M parametrów (odpowiednik GPT-2 Small)
            "small": cls(
                n_layer=12,
                n_head=12,
                n_embd=768,
                block_size=1024,
            ),
            # ~350M parametrów (odpowiednik GPT-2 Medium)
            "medium": cls(
                n_layer=24,
                n_head=16,
                n_embd=1024,
                block_size=1024,
            ),
        }
        if name not in presets:
            raise ValueError(f"Nieznany preset '{name}'. Dostępne: {list(presets.keys())}")
        return presets[name]


@dataclass
class TrainConfig:
    # Dane i ścieżki
    dataset_dir: str = "data"
    out_dir: str = "out"
    
    # Hiperparametry treningu
    batch_size: int = 12            # Rozmiar micro-batcha na krok
    gradient_accumulation_steps: int = 4  # Efektywny batch size = batch_size * grad_accum_steps
    max_iters: int = 5000           # Całkowita liczba kroków treningu
    
    # Optymalizacja
    learning_rate: float = 6e-4     # Maksymalna wartość learning rate
    min_lr: float = 6e-5            # Minimalny learning rate pod koniec schodzenia cosine
    warmup_iters: int = 200         # Liczba kroków rozgrzewki (warmup)
    lr_decay_iters: int = 5000      # Liczba kroków opadania cosine
    weight_decay: float = 1e-1      # Weight decay w AdamW
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0          # Przycinanie normy gradientu
    
    # Ewaluacja i logowanie
    eval_interval: int = 200        # Co ile kroków liczyć eval loss
    eval_iters: int = 50            # Ile batchy użyć do policzenia średniej eval loss
    log_interval: int = 10          # Co ile kroków wypisywać progress w konsoli
    sample_interval: int = 500      # Co ile kroków wygenerować tekst próbny
    always_save_checkpoint: bool = True
    
    # Sprzęt
    device: str = "cuda"            # 'cuda', 'cpu', 'mps'
    dtype: str = "bfloat16"         # 'bfloat16', 'float16', 'float32'
    compile: bool = False           # torch.compile (wymaga Triton na Linuxie)
