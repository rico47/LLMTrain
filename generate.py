"""
Skrypt do generowania tekstru (inferencji) na podstawie wytrenowanego modelu LLM.
"""

import os
import argparse
import torch
import tiktoken

from config import ModelConfig
from model import LLM


def generate_text(
    prompt: str,
    ckpt_path: str = "out/ckpt.pt",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cuda"
):
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"Urządzenie dla inferencji: {device}")

    # Wczytywanie tokenizatora tiktoken (gpt2)
    enc = tiktoken.get_encoding("gpt2")

    # Wczytywanie punktu kontrolnego
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Nie znaleziono pliku checkpointu: '{ckpt_path}'. Uruchom najpierw trening ('python train.py').")

    print(f"Wczytywanie checkpointu z: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = checkpoint['model_config']
    
    # Inicjalizacja modelu i wczytanie wag
    model = LLM(m_cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model wczytany pomyślnie! Liczba parametrów: {model.get_num_params():,}")
    print(f"\n--- PROMPT: ---\n{prompt}\n{'='*50}")

    # Kodowanie promptu
    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]

    # Autoregresywne generowanie tekstu
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    generated_text = enc.decode(y[0].tolist())
    print(f"\n--- WYGENEROWANY TEKST: ---\n{generated_text}\n{'='*50}")
    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generowanie tekstu przy użyciu wytrenowanego LLM")
    parser.add_argument('--prompt', type=str, default="ROSE:\nWhat is the matter, my lord?", help='Prompt startowy dla modelu')
    parser.add_argument('--ckpt', type=str, default="out/ckpt.pt", help='Ścieżka do checkpointu .pt')
    parser.add_argument('--max_tokens', type=int, default=200, help='Maksymalna liczba tokenów do wygenerowania')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperatura próbkowania (0.1 - 1.5)')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--device', type=str, default="cuda", help='Urządzenie (cuda/cpu)')
    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        ckpt_path=args.ckpt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )
