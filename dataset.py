"""
Moduł do wczytywania zbinaryzowanego zbioru danych dla treningu LLM.
"""

import os
import numpy as np
import torch


def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, device: str):
    """
    Szybkie losowanie minibatcza z pre-tokenizowanych plików binarnych (train.bin / val.bin).
    Wykorzystuje np.memmap do efektywnej pracy bez ładowania całego zbioru do RAM.
    """
    file_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Brak pliku danych '{file_path}'. Uruchom najpierw 'python prepare_data.py'.")

    # Wczytanie zmapowanej pamięci (uint16 dla słowników < 65536 tokenów)
    data = np.memmap(file_path, dtype=np.uint16, mode='r')

    # Losowanie indeksów startowych sekwencji
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Tworzenie tensorów wejściowych x oraz docelowych y (z przesunięciem o 1)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if 'cuda' in device:
        # Przypisanie do pamięci GPU bez użycia kserokopii stosu
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y
