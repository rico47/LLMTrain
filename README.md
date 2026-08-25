Aplikacja służy do wytrenowania własnego LLM
aby nastepnie mozna było ja użyc w np LMStudio należy ją przejonvertowac pakietem z llama.cpp do gpuf: convert_hf_to_gguf.py

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
