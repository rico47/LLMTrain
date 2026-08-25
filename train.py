"""
Skrypt treningowy dla modelu LLM.
Wspiera Mixed Precision (bfloat16/float16), akumulację gradientu, harmonogram Cosine LR z warmup,
oraz zapisywanie checkpointów.
"""

import os
import time
import math
import argparse
import numpy as np
import torch
import tiktoken

from config import ModelConfig, TrainConfig
from model import LLM
from dataset import get_batch


def get_lr(it: int, t_cfg: TrainConfig) -> float:
    """Oblicza wartość learning rate dla danego kroku it wg harmonogramu Cosine z Warmupem."""
    warmup_iters = min(t_cfg.warmup_iters, max(1, t_cfg.lr_decay_iters // 10))
    # 1) Rozgrzewka (Warmup)
    if it < warmup_iters:
        return t_cfg.learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) Po zakończeniu okresu opadania (lr_decay_iters)
    if it >= t_cfg.lr_decay_iters:
        return t_cfg.min_lr
    # 3) Opadanie Cosine pomiędzy warmup_iters a lr_decay_iters
    decay_ratio = (it - warmup_iters) / max(1, t_cfg.lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return t_cfg.min_lr + coeff * (t_cfg.learning_rate - t_cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: LLM, t_cfg: TrainConfig, m_cfg: ModelConfig, ctx):
    """Estymuje średnią stratę (loss) na zbiorze treningowym i walidacyjnym."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(t_cfg.eval_iters)
        for k in range(t_cfg.eval_iters):
            X, Y = get_batch(t_cfg.dataset_dir, split, t_cfg.batch_size, m_cfg.block_size, t_cfg.device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(m_cfg: ModelConfig, t_cfg: TrainConfig, resume: bool = False):
    os.makedirs(t_cfg.out_dir, exist_ok=True)
    
    # Wybór urządzenia i precyzji
    device = t_cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Uruchamianie treningu na urządzeniu: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM)")

    # Konfiguracja Mixed Precision (AMP)
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[t_cfg.dtype]
    ctx = torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=ptdtype)
    scaler = torch.amp.GradScaler('cuda', enabled=(t_cfg.dtype == 'float16'))

    # Wczytanie tokenizatora dla generowania próbki
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    # Inicjalizacja modelu
    iter_num = 0
    best_val_loss = 1e9

    ckpt_path = os.path.join(t_cfg.out_dir, 'ckpt.pt')
    if resume and os.path.exists(ckpt_path):
        print(f"Wczytywanie checkpointa z: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        m_cfg = checkpoint['model_config']
        model = LLM(m_cfg)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
        model.to(device)
        optimizer = model.configure_optimizers(t_cfg.weight_decay, t_cfg.learning_rate, (t_cfg.beta1, t_cfg.beta2), device)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Inicjalizacja nowego modelu...")
        model = LLM(m_cfg)
        model.to(device)
        optimizer = model.configure_optimizers(t_cfg.weight_decay, t_cfg.learning_rate, (t_cfg.beta1, t_cfg.beta2), device)

    # Opcjonalna kompilacja PyTorch 2.0
    if t_cfg.compile and hasattr(torch, 'compile'):
        print("Kompilowanie modelu (torch.compile)...")
        model = torch.compile(model)

    print(f"Całkowita liczba parametrów: {model.get_num_params():,}")

    # Główna pętla treningowa
    X, Y = get_batch(t_cfg.dataset_dir, 'train', t_cfg.batch_size, m_cfg.block_size, device)
    t0 = time.time()
    
    tokens_per_iter = t_cfg.batch_size * m_cfg.block_size * t_cfg.gradient_accumulation_steps
    print(f"Tokenów na iterację (efektywny batch): {tokens_per_iter:,}")

    while iter_num <= t_cfg.max_iters:
        # Obliczenie learning rate dla obecnego kroku
        lr = get_lr(iter_num, t_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 1. Okresowa ewaluacja i zapis punktu kontrolnego
        if iter_num % t_cfg.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, t_cfg, m_cfg, ctx)
            val_ppl = math.exp(losses['val']) if losses['val'] < 20 else float('inf')
            print(f"\n--- [EVAL Step {iter_num}/{t_cfg.max_iters}] Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Val Perplexity: {val_ppl:.2f} ---")
            
            if losses['val'] < best_val_loss or t_cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': m_cfg,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"Zapisywanie checkpointa do {ckpt_path} (Val Loss: {best_val_loss:.4f})")
                torch.save(checkpoint, ckpt_path)

        # 2. Okresowe generowanie próbki tekstu
        if iter_num % t_cfg.sample_interval == 0 and iter_num > 0 and enc is not None:
            model.eval()
            prompt = "FIRST SENATOR:\n"
            start_ids = enc.encode(prompt)
            x_sample = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]
            with torch.no_grad(), ctx:
                y_sample = model.generate(x_sample, max_new_tokens=100, temperature=0.8, top_k=40)
            sample_text = enc.decode(y_sample[0].tolist())
            print(f"\n>>> Wygenerowana próbka tekstu (krok {iter_num}):\n{sample_text}\n{'='*50}")
            model.train()

        # 3. Krok treningowy z akumulacją gradientów
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0

        for micro_step in range(t_cfg.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / t_cfg.gradient_accumulation_steps

            micro_loss += loss.item()
            X, Y = get_batch(t_cfg.dataset_dir, 'train', t_cfg.batch_size, m_cfg.block_size, device)
            scaler.scale(loss).backward()

        # Przycinanie gradientu
        if t_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)

        # Krok optymalizatora
        scaler.step(optimizer)
        scaler.update()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # 4. Logowanie wyników
        if iter_num % t_cfg.log_interval == 0:
            lossf = micro_loss * t_cfg.gradient_accumulation_steps
            tokens_per_sec = tokens_per_iter / dt
            print(f"Step {iter_num:5d}/{t_cfg.max_iters} | Loss: {lossf:.4f} | LR: {lr:.2e} | Time: {dt*1000:.1f}ms | Speed: {tokens_per_sec:.0f} tok/s")

        iter_num += 1

    print("\nTrening zakończony pomyślnie!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trening modelu LLM")
    parser.add_argument('--preset', type=str, default='micro', choices=['micro', 'mini', 'small', 'medium'], help='Preset konfiguracji modelu')
    parser.add_argument('--max_iters', type=int, default=1000, help='Maksymalna liczba iteracji')
    parser.add_argument('--batch_size', type=int, default=12, help='Rozmiar micro-batcha')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Katalog z danymi')
    parser.add_argument('--out_dir', type=str, default='out', help='Katalog z wynikami i checkpointami')
    parser.add_argument('--resume', action='store_true', help='Wznowienie treningu z checkpointu')
    args = parser.parse_args()

    m_config = ModelConfig.get_preset(args.preset)
    t_config = TrainConfig(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        lr_decay_iters=args.max_iters,
        learning_rate=args.lr,
    )

    train(m_config, t_config, resume=args.resume)
