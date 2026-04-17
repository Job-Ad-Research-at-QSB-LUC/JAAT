import torch
import nltk
import re
from torch.utils.data import Dataset

CLEAN_PATTERN = re.compile(r"(;|\s\+\s|\s\*\s|\s\-\s|\s•\s|\s·\s|--|\*\*)")

def get_device_settings():
    if torch.cuda.is_available():
        return "cuda", 2048
    return "cpu", 64

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def sent_tokenize(text):
    text = ". ".join(text.split("\n"))
    text = CLEAN_PATTERN.sub(". ", text)
    text = text.replace("\n", ".").replace("..", ".")
    return nltk.sent_tokenize(text)