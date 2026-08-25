"""
Skrypt pobierający przykładowy zestaw danych tekstowych i wykonujący tokenizację BPE (tiktoken)
do natywnych plików binarnych numpy dla treningu LLM.
"""

import os
import argparse
import urllib.request
import numpy as np
import tiktoken


def prepare_dataset(input_file: str = None, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Pobieranie domyślnego pliku tekstowego jeśli brak pliku wejściowego
    if input_file is None or not os.path.exists(input_file):
        print("Brak wskazanego pliku tekstowego. Pobieranie przykładowego zbioru 'Tiny Shakespeare'...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        input_file = os.path.join(out_dir, 'input.txt')
        if not os.path.exists(input_file):
            urllib.request.urlretrieve(data_url, input_file)
            print(f"Pobrano plik do: {input_file}")

    print(f"Wczytywanie tekstu z: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    n_chars = len(data)
    print(f"Długość korpusu tekstowego: {n_chars:,} znaków")

    # 2. Tokenizacja BPE za pomocą tiktoken (GPT-2 encoding)
    print("Rozpoczynanie tokenizacji przy użyciu tiktoken (gpt2)...")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(data)
    print(f"Liczba wygenerowanych tokenów: {len(ids):,}")

    # 3. Podział na zbiór treningowy (90%) i walidacyjny (10%)
    n = len(ids)
    train_ids = ids[:int(n*0.9)]
    val_ids = ids[int(n*0.9):]

    # Przekształcenie w tablice uint16 (słownik gpt2 wynosi 50,257, mieści się w uint16 0-65535)
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_file = os.path.join(out_dir, 'train.bin')
    val_file = os.path.join(out_dir, 'val.bin')

    train_ids.tofile(train_file)
    val_ids.tofile(val_file)

    print(f"Sukces! Zapisano:")
    print(f" - Train dataset: {train_file} ({len(train_ids):,} tokenów, {os.path.getsize(train_file)/(1024*1024):.2f} MB)")
    print(f" - Val dataset:   {val_file} ({len(val_ids):,} tokenów, {os.path.getsize(val_file)/(1024*1024):.2f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Przygotowanie zbioru danych dla LLM")
    parser.add_argument('--input', type=str, default=None, help='Ścieżka do własnego pliku .txt (opcjonalne)')
    parser.add_argument('--out_dir', type=str, default='data', help='Katalog docelowy na pliki binarne')
    args = parser.parse_args()

    prepare_dataset(input_file=args.input, out_dir=args.out_dir)
