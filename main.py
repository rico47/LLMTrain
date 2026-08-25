"""
Główny punkt wejściowy dla programu trenującego i uruchamiającego model LLM.
"""

import sys
import argparse

from prepare_data import prepare_dataset
from train import train
from generate import generate_text
from config import ModelConfig, TrainConfig


def main():
    parser = argparse.ArgumentParser(
        description="Program do trenowania i uruchamiania własnego modelu LLM w PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  1. Przygotowanie danych (pobranie i tokenizacja):
     python main.py prepare

  2. Trening modelu (preset 'micro' ~10M parametrów):
     python main.py train --preset micro --iters 1000

  3. Generowanie tekstu z wytrenowanego checkpointa:
     python main.py generate --prompt "To be or not to be"

  4. Pełny cykl (prepare + train + generate):
     python main.py demo
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Dostępne polecenia")

    # Command: prepare
    prepare_parser = subparsers.add_parser("prepare", help="Pobiera zbiór danych i dokonuje tokenizacji BPE")
    prepare_parser.add_argument("--input", type=str, default=None, help="Ścieżka do własnego pliku tekstowego .txt")
    prepare_parser.add_argument("--out_dir", type=str, default="data", help="Katalog wyjściowy z danymi")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Uruchamia trening modelu LLM")
    train_parser.add_argument("--preset", type=str, default="micro", choices=["micro", "mini", "small", "medium"], help="Preset rozmiaru modelu")
    train_parser.add_argument("--iters", type=int, default=1000, help="Liczba kroków (iteracji) treningowych")
    train_parser.add_argument("--batch_size", type=int, default=12, help="Rozmiar micro-batcha na GPU")
    train_parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    train_parser.add_argument("--data_dir", type=str, default="data", help="Katalog z zbinaryzowanymi danymi")
    train_parser.add_argument("--out_dir", type=str, default="out", help="Katalog zapisu checkpointów")
    train_parser.add_argument("--resume", action="store_true", help="Wznowienie z istniejącego checkpointu")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generuje tekst z wytrenowanego modelu")
    gen_parser.add_argument("--prompt", type=str, default="FIRST SENATOR:\nWhat is your sentence?", help="Tekst początkowy (prompt)")
    gen_parser.add_argument("--ckpt", type=str, default="out/ckpt.pt", help="Ścieżka do pliku checkpointu .pt")
    gen_parser.add_argument("--max_tokens", type=int, default=200, help="Maksymalna liczba nowych tokenów")
    gen_parser.add_argument("--temp", type=float, default=0.8, help="Temperatura próbkowania")
    gen_parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")

    # Command: demo
    demo_parser = subparsers.add_parser("demo", help="Wykonuje pełny automatyczny demonstracyjny cykl: prepare -> train (200 iters) -> generate")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset(input_file=args.input, out_dir=args.out_dir)

    elif args.command == "train":
        m_cfg = ModelConfig.get_preset(args.preset)
        t_cfg = TrainConfig(
            dataset_dir=args.data_dir,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            max_iters=args.iters,
            lr_decay_iters=args.iters,
            learning_rate=args.lr,
        )
        train(m_cfg, t_cfg, resume=args.resume)

    elif args.command == "generate":
        generate_text(
            prompt=args.prompt,
            ckpt_path=args.ckpt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
        )

    elif args.command == "demo":
        print("=== [KROK 1/3] Przygotowanie danych ===")
        prepare_dataset(out_dir="data")
        
        print("\n=== [KROK 2/3] Trening modelu (200 iteracji) ===")
        m_cfg = ModelConfig.get_preset("micro")
        t_cfg = TrainConfig(
            dataset_dir="data",
            out_dir="out",
            batch_size=8,
            max_iters=200,
            lr_decay_iters=200,
            eval_interval=100,
            log_interval=20,
            sample_interval=100,
        )
        train(m_cfg, t_cfg)

        print("\n=== [KROK 3/3] Generowanie próbki tekstu ===")
        generate_text(
            prompt="KING RICHARD III:\nGive me another horse: bind up my wounds.",
            ckpt_path="out/ckpt.pt",
            max_new_tokens=150,
            temperature=0.8,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
