from typing import List, Union, Tuple, Dict, Optional
import pandas as pd
import torch
import multiprocessing as mp
from sentence_transformers import SentenceTransformer, util
import importlib_resources as impresources
import json
import re
import compress_pickle
from operator import itemgetter
from functools import partial

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
import requests

from .base import get_device_settings
from .config import logger
from .utils import progress_bar

CLEAN_RE = re.compile(r'[^a-z0-9]+')
    
def get_context(text: str, keywords: List[str], n: int) -> Optional[List[str]]:
    if not text:
        return None
    
    text = CLEAN_RE.sub(' ', text.lower())
    words = text.split()
    k_set = {k.strip() for k in keywords}
    found_indices = [i for i, w in enumerate(words) if any(k in w for k in k_set)]
    
    if not found_indices:
        return None
        
    return [" ".join(words[max(0, idx-n):min(idx+n+1, len(words))]) for idx in found_indices]
    
clf = None

def init_pool(classifier):
    global clf
    clf = classifier

def classify(contexts: List[Union[str, None]]) -> int:
    global clf
    if contexts is None:
        return 0
    res = clf.predict(contexts)
    return 1 if sum(res) > 0 else 0

class CREAM():
    def __init__(self, keywords: List[str], rules: List[Tuple[str, any]], class_name: str = "label", n: int = 4, threshold: float = 0.9):
        if not isinstance(keywords, list) or not isinstance(rules, list):
            logger.error("ERROR: keywords and rules must be given as a list of values. \n KEYWORDS: [k1, k2, ..., kn] \n RULES: [(rule_1, label), (rule_2, label), ..., (rule_n, label)]")
            return
        
        logger.info("Initalizing CREAM...")
        self.device, _ = get_device_settings()

        self.class_name = class_name
        self.n = n
        self.threshold = threshold
        self.keywords = keywords
        rules = [(k, 0) for k in self.keywords] + rules
        self.rules = pd.DataFrame(rules, columns=["rule", self.class_name])
        self.rule_texts = self.rules['rule'].tolist()

        self.model = SentenceTransformer("thenlper/gte-large", device=self.device)

        self.encoded_rules = self.model.encode(self.rule_texts)
        self.encoded_rules = torch.nn.functional.normalize(self.encoded_rules, p=2, dim=1)
        self.rule_map = dict(zip(self.rule_texts, self.rules[self.class_name].tolist()))

        logger.info("Finished.")

    def get_sim(self, q: str) -> Dict:
        sim_scores = util.cos_sim(self.model.encode([q]), self.encoded_rules)
        return dict(zip(self.rule_map.keys(), sim_scores[0].tolist()))
    
    def label_from_max(self, scores: Dict) -> Tuple[str, str, float]:
        max_rule = max(scores, key=scores.get)
        label = self.rule_map[max_rule]
        return max_rule, label, scores[max_rule]
        
    def run(self, texts: List[str]) -> pd.DataFrame:
        all_contexts = []
        text_indices = []
        
        for i, txt in enumerate(texts):
            ctxs = get_context(txt, self.keywords, self.n)
            if ctxs:
                all_contexts.extend(ctxs)
                text_indices.extend([i] * len(ctxs))
        
        if not all_contexts:
            return pd.DataFrame({"text": texts, "inferred_rule": None, "inferred_label": 0, "inferred_confidence": None})

        embeddings = self.model.encode(all_contexts, convert_to_tensor=True, show_progress_bar=True)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        sim_matrix = torch.mm(embeddings, self.encoded_rules.T)
        
        max_scores, max_indices = torch.max(sim_matrix, dim=1)
        max_scores = max_scores.cpu().tolist()
        max_indices = max_indices.cpu().tolist()

        results = [ (None, 0, 0.0) ] * len(texts)
        
        for idx, score, rule_idx in zip(text_indices, max_scores, max_indices):
            if score >= self.threshold:
                if score > results[idx][2]:
                    results[idx] = (self.rule_texts[rule_idx], self.rule_labels[rule_idx], round(score, 3))

        df = pd.DataFrame(texts, columns=["text"])
        df['inferred_rule'], df['inferred_label'], df['inferred_confidence'] = zip(*results)
        return df
    
class JobTag():
    def __init__(self, class_name: str, n: int = 4) -> None:
        with open(impresources.files("JAAT.data") / "keywords.json", 'r') as f:
            self.keywords = json.load(f)

        self.classes = sorted([x for x in self.keywords])

        if class_name not in self.classes:
            logger.error("Usage Error: please select one of the available classes:\n[{}]".format(" | ".join(self.classes)))
            return
        
        self.class_name = class_name
        try:
            local_path = hf_hub_download(
                repo_id="loyoladatamining/JobTag",
                filename="v1/{}.lzma".format(self.class_name)
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logger.info("No internet connection detected. Attempting to load from cache...")
            try:
                return hf_hub_download(
                    repo_id="loyoladatamining/JobTag", 
                    filename="v1/{}.lzma".format(self.class_name), 
                    local_files_only=True
                )
            except LocalEntryNotFoundError:
                raise RuntimeError("{} is not cached and no internet connection is available.".format(self.class_name))
            
        with open(local_path, 'rb') as f:
            self.clf = compress_pickle.load(f, compression="lzma", set_default_extension=False)
        self.n = n

    def get_tag(self, text: str) -> Tuple[str, int]:
        contexts = get_context(text, keywords=self.keywords[self.class_name], n=self.n)
        if contexts is None:
            return (self.class_name, 0)
        res = self.clf.predict(contexts)
        return (self.class_name, 1 if sum(res) > 0 else 0)
    
    def get_tag_batch(self, texts: List[str]) -> List[int]:
        with mp.Pool(mp.cpu_count(), initializer=init_pool, initargs=(self.clf,)) as pool:
            c = pool.imap(partial(get_context, keywords=self.keywords[self.class_name], n=self.n), texts)
            all_contexts = list(c)
            p = progress_bar(pool.imap(classify, all_contexts), total=len(all_contexts))
            res = list(p)
            pool.close()
        return res