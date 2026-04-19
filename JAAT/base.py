from typing import List, Tuple, Any
import torch
import nltk
import re
from torch.utils.data import Dataset
import logging

CLEAN_PATTERN = re.compile(r"(;|\s\+\s|\s\*\s|\s\-\s|\s•\s|\s·\s|--|\*\*)")

logger = logging.getLogger("JAAT")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def get_device_settings() -> Tuple[str, int]:
    if torch.cuda.is_available():
        return "cuda", 2048
    return "cpu", 64

class ListDataset(Dataset):
    def __init__(self, original_list: List[Any]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> Any:
        return self.original_list[i]

def sent_tokenize(text: str) -> List[str]:
    text = ". ".join(text.split("\n"))
    text = CLEAN_PATTERN.sub(". ", text)
    text = text.replace("\n", ".").replace("..", ".")
    return nltk.sent_tokenize(text)