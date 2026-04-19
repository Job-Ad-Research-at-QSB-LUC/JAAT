from typing import List, Optional
import pandas as pd
import multiprocessing as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import importlib_resources as impresources
import json
import string
import re
from collections import Counter
import nltk
import syllables
import ahocorasick

from .base import ListDataset, sent_tokenize, get_device_settings
from .config import logger
from .utils import progress_bar

class StdName():
    sub_dict = None
   
    def __init__(self) -> None:
        with open(impresources.files("JAAT.data") / "sub_dict.json", 'r') as f:
            self.sub_dict = json.load(f)

        self.automaton = ahocorasick.Automaton()
        for k, v in self.sub_dict.items():
            self.automaton.add_word(k, (k, v))
        self.automaton.make_automaton()

    def standardize(self, text: str) -> str:
        if pd.isnull(text) == True:
            return " "

        text = " " + text.lower() + " "

        for _, (orig, replacement) in self.automaton.iter(text):
            text = text.replace(f" {orig.lower()} ", f" {replacement.lower()} ")

        return " ".join(text.split()).upper()
    
class FirmExtract():
    STD = None
    CAMEL_RE = re.compile(r'([a-z0-9])([A-Z])')
    CLEAN_RE_1 = re.compile(r'\<.*$')
    CLEAN_RE_2 = re.compile(r'\'\w+')

    def __init__(self, standardize: bool = True) -> None:
        self.device, _ = get_device_settings()

        model = AutoModelForTokenClassification.from_pretrained("loyoladatamining/firmNER-v3", id2label={0: 'O', 1: 'B-ORG', 2: 'I-ORG'}, label2id={'O': 0, 'B-ORG': 1, 'I-ORG': 2})
        tokenizer = AutoTokenizer.from_pretrained("loyoladatamining/firmNER-v3", model_max_length=1024, max_length=1024, truncation=True)
        self.pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device, aggregation_strategy="max")

        remove = string.punctuation
        remove = remove.replace(".", "").replace(",", "").replace("-", "").replace("&", "").replace("'","")
        self.pattern = r"[{}]".format(remove)

        self.standardize = standardize
        if self.standardize == True:
            self.STD = StdName()

    def clean_company(self, company: str) -> str:
        company = company.replace('\\', '')

        company = self.CLEAN_RE_1.sub('', company)
        company = self.CLEAN_RE_2.sub('', company)
        company = self.pattern.sub("", company)
        company = company.replace("\'", "").replace(" ’", "’")

        if len(company) > 0 and company[-1] == ',':
            company = company[:-1]

        company = " ".join(company.split())
        return company
    
    def split_words(self, text: str) -> str:
        return self.CAMEL_RE.sub(r'\1 \2', text)

    def extract_firm(self, tagged: List[dict], return_one: bool = False, return_score: bool = False):
        cands = []
        for r in tagged:
            if r["entity_group"] == "ORG":
                cands.append((self.split_words(r["word"]), r["score"]))

        if len(cands) > 0:
            company = sorted(cands, key=lambda x:x[1], reverse=True)
            scores = [x[1] for x in company]
            company = [self.clean_company(x[0]) for x in company]
        else:
            if return_score == True and return_one == True:
                return None, 0
            else:
                return None

        if all(len(x) == 0 for x in company) or company[0] in ["Inc", "Inc."]:
            if return_score == True and return_one == True:
                return None, 0
            else:
                return None
        
        if self.standardize == True:
            company = [self.STD.standardize(x) for x in company]

        if return_one == True:
            ret = company[0]
            score = scores[0]
        else:
            ret = set(company)

        if return_score == True and return_one == True:
            return ret, score
        else:
            return ret

    def get_firm(self, text, return_one=True, return_score=True):
        tagged = self.pipe(text)
        return self.extract_firm(tagged, return_one, return_score)
    
    def get_firm_batch(self, texts, return_one=True, return_score=True):
        batch = ListDataset(texts)
        results = []
        for r in progress_bar(self.pipe(batch), total=len(texts)):
            results.append(self.extract_firm(r, return_one, return_score))
        return results
    
class WageExtract():
    def __init__(self) -> None:
        self.device, self.batch_size = get_device_settings()

        self.keywords = [
            '$', 'dollar', 'dollars',
            'salary', 'salaries',
            'wage', 'wages',
            'compensation', 'compensations',
            'annual', 'yearly', 'peryear', 'perannum',
            'hourly', 'perhour'
        ]

        self.detok = nltk.treebank.TreebankWordDetokenizer()

        model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/is_pay")
        tokenizer = AutoTokenizer.from_pretrained(
            "loyoladatamining/is_pay",
            use_fast=True,
            max_length=64,
            truncation=True
        )
        self.ispay_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, max_length=64, device=self.device, truncation=True, batch_size=self.batch_size, num_workers=mp.cpu_count())

        model = model = AutoModelForTokenClassification.from_pretrained("loyoladatamining/wage-ner-v2", max_length=128, id2label={0:'O', 1:'B-MIN', 2:'B-MAX'}, label2id={'O': 0, 'B-MIN': 1, 'B-MAX': 2})
        tokenizer = AutoTokenizer.from_pretrained(
            "loyoladatamining/wage-ner-v2",
            max_length=128,
            model_max_length=128,
            truncation=True
        )
        self.ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device, aggregation_strategy="simple")

        model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/pay-freq-v2")
        tokenizer = AutoTokenizer.from_pretrained(
            "loyoladatamining/pay-freq-v2", 
            max_length=128,
            truncation=True
        )
        self.freq_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, max_length=128, device=self.device, truncation=True)

    def preproc(self, text: str) -> str:
        text = text.replace("401k", "")
        text = re.sub(r"(\.[0-9])0{3,}", r"\g<1>0", text)
        text = re.sub(r"([0-9])(to)([0-9])", r"\1 \2 \3", text)
        text = re.sub(r"([0-9]{1,3})k", r"\g<1>000", text)
        text = re.sub(r"((salary)|(range)|(pay)|(rate)|(up to))(:? )(\d)", r"\1\6$\7", text)
        text = re.sub(r"((Salary)|(Range)|(Pay)|(Rate))(:? )(\d)", r"\1\6$\7", text)
        text = re.sub(r"usd ?(\d+)", r"$\1", text)
        text = re.sub(r"USD ?(\d+)", r"$\1", text)
        text = re.sub(r"(\d+)(/)(hr)", r"\1 \2 \3", text)
        text = re.sub(r"(\d+)([-/])(\w+)", r"\1 \2 \3", text).replace ("*", " ").replace("\\.",".")
        text = " ".join(text.split())
        return text

    def get_largest_chunk(self, text: str , n: int = 6) -> str:
        text = text.replace("per hour", "perhour").replace("per year", "peryear").replace("per annum", "perannum")
        tokens = nltk.word_tokenize(text)
        idx = [i for i, x in enumerate(tokens) if any(y in x.lower() for y in self.keywords)]
        if len(idx) == 0:
            return None   
        mi = min(idx)
        ma = max(idx)
        temp = tokens[max(mi-n, 0):min(ma+1+n, len(tokens)-1)]
        ret = self.detok.detokenize(temp)
        ret = ret.replace("perhour", "per hour").replace("peryear", "per year").replace("perannum", "per annum")
        return ret
    
    def extract_wage(self, predict: List[dict]) -> dict:
        start = 0
        MIN = ""
        for p in predict:
            if p["entity_group"] == "MIN":
                if start == 0 or p["start"] == start:
                    MIN += p["word"]
                    start = p["end"]
                else:
                    break
        start = 0
        MAX = ""
        for p in predict:
            if p["entity_group"] == "MAX":
                if start == 0 or p["start"] == start:
                    MAX = MAX + p["word"]
                    start = p["end"]
                else:
                    break
        return {"min":MIN.replace("$",""), "max":MAX.replace("$","")}

    def get_wage(self, text: str) -> dict:
        sentences = sent_tokenize(text)
        is_pay = self.ispay_pipe(sentences)
        if all(x["label"] == "LABEL_0" for x in is_pay):
            return "The provided text does not contain a wage statement."
        
        text = self.get_largest_chunk(self.preproc(text))
        pred = self.extract_wage(self.ner_pipe(text))
        freq = self.freq_pipe(text)[0]["label"]
        pred["frequency"] = freq
        return pred

    def get_wage_batch(self, texts: List[str]) -> List[dict]:
        temp = []
        for i, x in enumerate(texts):
            temp.extend([(i, y) for y in sent_tokenize(x)])
        all_indices = [x[0] for x in temp]
        all_sentences = [x[1] for x in temp]

        batch = ListDataset(all_sentences)
        is_pay = []
        logger.info("Predicting is_pay...")
        for p in progress_bar(self.ispay_pipe(batch), total=len(all_sentences)):
            if p["label"] == "LABEL_1":
                is_pay.append(1)
            else:
                is_pay.append(0)

        res = list(zip(all_indices, is_pay))
        counts = Counter()
        for r in res:
            counts[r[0]] += r[1]

        indices = []
        cands = []
        for c in counts:
            if counts[c] > 0:
                check = self.get_largest_chunk(self.preproc(texts[c]))
                if check is not None:
                    indices.append(c)
                    cands.append(check)

        logger.info("{} / {} contain wage statements.\n".format(len(indices), len(texts)))

        new_batch = ListDataset(cands)

        logger.info("Extracting wage...")
        preds = []
        for p in progress_bar(self.ner_pipe(new_batch), total=len(new_batch)):
            preds.append(self.extract_wage(p))
        
        logger.info("\nPredicting pay frequency...")
        freqs = []
        for p in progress_bar(self.freq_pipe(new_batch), total=len(new_batch)):
            freqs.append(p["label"])

        final = []
        for x, y in zip(preds, freqs):
            if x["min"] == "" and x["max"] == "":
                final.append(None)
            else:
                x["frequency"] = y
                final.append(x)
    
        ret = []
        count = 0
        for i, x in enumerate(texts):
            if i in indices:
                ret.append(final[count])
                count += 1
            else:
                ret.append(None)

        assert len(ret) == len(texts)

        total = sum([1 for x in ret if x is not None])
        logger.info("\nSummary: found {} wage statements.".format(total))

        return ret
    
class Readability():
    def __init__(self) -> None:
        self.PUNCT = set(string.punctuation)

    def fk(self, text: str) -> Optional[float]:
        if text == "":
            return None

        tokens = nltk.word_tokenize(text)
        words = len(tokens)
        sentences = len(nltk.sent_tokenize(text))
        syl = sum([syllables.estimate(x) for x in tokens if x not in self.PUNCT])
        return round(206.835 - (1.015 * (words / sentences)) - (84.6 * (syl / words)), 2)
    
    def get_readability(self, text: str) -> Optional[float]:
        return self.fk(text)
    
    def get_readability_batch(self, texts: List[str]) -> List[Optional[float]]:
        scores = []
        with mp.Pool(int(mp.cpu_count() / 2)) as pool:
            for res in progress_bar(pool.imap(self.fk, texts), total=len(texts)):
                scores.append(res)
        return scores
