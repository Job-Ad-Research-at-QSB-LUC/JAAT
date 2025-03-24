from transformers import pipeline
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import nltk
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import Dataset
import importlib_resources as impresources
import multiprocessing as mp
import re
import string
from operator import itemgetter
from pathlib import Path
import json
from functools import partial
import compress_pickle
from collections import Counter

tqdm.pandas()

def sent_tokenize(text):
    text = text.replace("\n", ".").replace("..", ".")
    return nltk.sent_tokenize(text)

def init_pool(classifier):
    global clf
    clf = classifier

def classify(contexts):
    global clf
    if contexts is None:
        return 0
    res = clf.predict(contexts)
    return 1 if sum(res) > 0 else 0

def get_context(text, keywords, n):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)

    words = text.split()
    found_index = [i for i, w in enumerate(words) if any(k.strip() in w for k in keywords)]
    context = [" ".join(words[max(0, idx-n):min(idx+1, len(words))]) for idx in found_index]

    if len(context) > 0:
        return context
    else:
        return None

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

class StdName():
    sub_dict = None
   
    def __init__(self):
        with open(impresources.files("JAAT.data") / "sub_dict.json", 'r') as f:
            self.sub_dict = json.load(f)

    def standardize(self, text):
        text = " " + text + " "

        for k, v in self.sub_dict.items():
            if len(k) == 1 and (v == "" or v == " "):
                text = text.replace(k, v)
            else:
                text = text.replace(" "+k+" ", " "+v+" ")

        return " ".join(text.split())

class TaskMatch():
    def __init__(self, threshold=0.9):
        print("INIT", flush=True)
        if torch.cuda.is_available() == True:
            self.device = "cuda"
            self.batch_size = 2048
        else:
            self.device = "cpu"
            self.batch_size = 64

        self.threshold = threshold

        print("Preparing embeddings...", flush=True)
        self.embedding_model = SentenceTransformer("thenlper/gte-small", device=self.device)
        tasks = pd.read_csv(impresources.files("JAAT.data") / "Task_DWA.csv")[["Task ID", "Task"]].drop_duplicates()
        self.tasks = tasks.reset_index().drop("index", axis=1)
        self.task_embed = self.embedding_model.encode(tasks.Task.to_list(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.task_embed = self.task_embed.to(self.device)

        print("Setting up pipeline...", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/task-classifier-mini-improved2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "loyoladatamining/task-classifier-mini-improved2",
            use_fast=True,
            max_length=64,
            truncation=True
        )
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, max_length=64, device=self.device, truncation=True, batch_size=self.batch_size, num_workers=mp.cpu_count())
        print("Finished.", flush=True)

    def get_candidates(self, text):
        text = ". ".join(text.split("\n"))
        text = text.replace(";", ".").replace(" + ", ". ").replace(" * ", ". ").replace(" - ", ". ")
        s = sent_tokenize(text.strip())
        all_data = [ss for ss in s if len(ss.split()) <= 48]

        positive = []
        predictions = []

        temp = ListDataset(all_data)
        for r in self.pipe(temp):
            predictions.append(r)

        count = 0
        for x, y in zip(all_data, predictions):
            if y['label'] == 'LABEL_1':
                positive.append(x)
                count += 1
        return positive
    
    def get_candidates_batch(self, texts):
        all_data = []
        for i, t in enumerate(texts):
            t = ". ".join(t.split("\n"))
            t = t.replace(";", ".").replace(" + ", ". ").replace(" * ", ". ").replace(" - ", ". ")
            s = sent_tokenize(t.strip())
            all_data.extend([(i, ss) for ss in s if len(ss.split()) <= 48])

        positive = []
        predictions = []

        temp = ListDataset([x[1] for x in all_data])
        for r in self.pipe(temp):
            predictions.append(r)

        count = 0
        for x, y in zip(all_data, predictions):
            if y['label'] == 'LABEL_1':
                positive.append(x)
                count += 1
        return positive

    def get_tasks(self, text):
        positive = self.get_candidates(text)
        if len(positive) == 0:
            return []

        q_embed = self.embedding_model.encode(positive, convert_to_tensor=True, batch_size=64)
        if self.device == "cuda":
            q_embed = q_embed.to(self.device)

        search = util.semantic_search(corpus_embeddings=self.task_embed, query_embeddings=q_embed, top_k=1)

        found = 0
        matched_tasks = []
        for x in search:
            if x[0]["score"] >= self.threshold:
                found += 1
                matched_tasks.append((str(self.tasks.iloc[x[0]["corpus_id"]]["Task ID"]), self.tasks.iloc[x[0]["corpus_id"]]["Task"]))

        return matched_tasks
    
    def get_tasks_batch(self, texts):
        all_data = self.get_candidates_batch(texts)

        q_embed = self.embedding_model.encode([x[1] for x in all_data], convert_to_tensor=True, batch_size=64)
        if self.device == "cuda":
            q_embed = q_embed.to(self.device)

        search = util.semantic_search(corpus_embeddings=self.task_embed, query_embeddings=q_embed, top_k=1)

        found = 0
        matched_tasks = []
        for _ in range(len(texts)):
            matched_tasks.append([])
        for x, y in zip(search, all_data):
            if x[0]["score"] >= self.threshold:
                found += 1
                matched_tasks[y[0]].append((str(self.tasks.iloc[x[0]["corpus_id"]]["Task ID"]), self.tasks.iloc[x[0]["corpus_id"]]["Task"]))

        return matched_tasks
    
class TitleMatch():

    def __init__(self, batch_size=16):
        print("INIT", flush=True)
        if torch.cuda.is_available() == True:
            self.device = "cuda"
            self.batch_size = 2048
        else:
            self.device = "cpu"
            self.batch_size = 64

        print("Loading data...", flush=True)
        titles = pd.read_csv(impresources.files("JAAT.data") / "titles_v2.csv")[["Reported Job Title", "O*NET-SOC Code"]].drop_duplicates(["Reported Job Title"]).dropna()
        titles.columns = ["title", "code"]
        self.titles = titles.reset_index().drop("index", axis=1)

        print("Preparing embeddings...", flush=True)
        self.embedding_model = SentenceTransformer("thenlper/gte-small", device=self.device)
        self.title_embed = self.embedding_model.encode(self.titles.title.to_list(), convert_to_tensor=True, show_progress_bar=True)
        self.title_embed = self.title_embed.to(self.device)

        print("Loading title models...", flush=True)
        self.value_model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/title_value", num_labels=1, problem_type="regression")
        self.value_tokenizer = AutoTokenizer.from_pretrained("loyoladatamining/title_value")
        self.value_pipe = pipeline("text-classification", model=self.value_model, tokenizer=self.value_tokenizer, device=self.device, function_to_apply="none")

        self.feature_model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/title_feature").to(self.device)
        self.feature_tokenizer = AutoTokenizer.from_pretrained("loyoladatamining/title_feature")
        self.feature_batch = batch_size

        print("Finished.")

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_title(self, text):
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, list):
            pass
        else:
            print("Error: input must be string or a list of strings.")
            return
        
        print("Extracting title values...", flush=True)
        values = []
        temp = ListDataset(text)
        for res in tqdm(self.value_pipe(temp, batch_size=self.feature_batch), total=len(text)):
            values.append(round(res["score"], 1))

        print("Extracting title features...", flush=True)
        features = []
        batches = self.batch(text, n=self.feature_batch)
        batches = ListDataset(list(batches))
        for b in tqdm(batches, total=len(batches)):
            inputs = self.feature_tokenizer.batch_encode_plus(b, truncation=True, max_length=32, padding="max_length", return_tensors="pt").to("cuda")
            outputs = self.feature_model(inputs.input_ids)
            logits = outputs.logits
            with torch.no_grad():
                for l in logits:
                    res = (self.sigmoid(l.cpu()) > 0.98).numpy().astype(int).reshape(-1)
                    temp = sorted([self.feature_model.config.id2label[x] for x in np.where(res == 1)[0]])
                    if "none" in temp and len(temp) > 1:
                        temp = [x for x in temp if x != "none"]
                    features.append(";".join(temp))
  
        print("Matching titles to codes...", flush=True)
        q_embed = self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=True)
        q_embed = q_embed.to(self.device)
        
        search = util.semantic_search(corpus_embeddings=self.title_embed, query_embeddings=q_embed, top_k=1)

        results = []
        for s, v, f in zip(search, values, features):
            results.append((self.titles.title[s[0]["corpus_id"]], self.titles.code[s[0]["corpus_id"]], round(s[0]["score"], 3), v, f))

        return results
    
class FirmExtract():
    STD = None

    def __init__(self, standardize=False):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"
        model = AutoModelForTokenClassification.from_pretrained("loyoladatamining/firmNER-v2-small", id2label={0: 'O', 1: 'B-ORG', 2: 'I-ORG'}, label2id={'O': 0, 'B-ORG': 1, 'I-ORG': 2})
        tokenizer = AutoTokenizer.from_pretrained("loyoladatamining/firmNER-v2-small")
        self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=self.device, aggregation_strategy="max")

        remove = string.punctuation
        remove = remove.replace(".", "").replace(",", "").replace("-", "").replace("&", "").replace("'","")
        self.pattern = r"[{}]".format(remove)

        self.standardize = standardize
        if self.standardize == True:
            self.STD = StdName()

    def clean_company(self, company):
        company = company.replace('\\', '')

        company = re.sub(r'\<.*$', '', company)
        company = re.sub(r'\'\w+', '', company)
        company = re.sub(self.pattern, "", company) 
        company = company.replace("\'", "")

        if len(company) > 0 and company[-1] == ',':
            company = company[:-1]

        company = " ".join(company.split())
        return company

    def extract_firm(self, tagged):
        cands = []
        for r in tagged:
            if r["entity_group"] == "ORG":
                cands.append((r["word"], r["score"]))

        if len(cands) > 0:
            company = sorted(cands, key=lambda x:x[1], reverse=True)
            company = [self.clean_company(x[0]) for x in cands]
        else:
            return None
        
        if self.standardize == True:
            company = [self.STD.standardize(x) for x in company]

        return set(company)

    def get_firm(self, text):
        tagged = self.pipe(text)
        return self.extract_firm(tagged)
    
    def get_firm_batch(self, texts):
        batch = ListDataset(texts)
        results = []
        for r in tqdm(self.pipe(batch), total=len(texts)):
            results.append(self.extract_firm(r))
        return results

class CREAM():
    def __init__(self, keywords, rules, class_name="label", n=4, threshold=0.9):
        if not isinstance(keywords, list) or not isinstance(rules, list):
            print("ERROR: keywords and rules must be given as a list of values. \n KEYWORDS: [k1, k2, ..., kn] \n RULES: [(rule_1, label), (rule_2, label), ..., (rule_n, label)]")
            return
        
        print("INIT", flush=True)
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.class_name = class_name
        self.n = n
        self.threshold = threshold
        self.keywords = keywords
        rules = [(k, 0) for k in self.keywords] + rules
        self.rules = pd.DataFrame(rules, columns=["rule", self.class_name])

        self.model = SentenceTransformer("thenlper/gte-large", device=self.device)

        self.encoded_rules = self.model.encode(self.rules['rule'].tolist())
        self.rule_map = dict(zip(self.rules['rule'].tolist(), self.rules[self.class_name].tolist()))

        print("Finished.", flush=True)

    def get_sim(self, q):
        sim_scores = util.cos_sim(self.model.encode([q]), self.encoded_rules)
        return dict(zip(self.rule_map.keys(), sim_scores[0].tolist()))
    
    def label_from_max(self, scores):
        max_rule = max(scores, key=scores.get)
        label = self.rule_map[max_rule]
        return max_rule, label, scores[max_rule]
    
    def get_context(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', ' ', text)

        words = text.split()
        found_index = [i for i, w in enumerate(words) if any(k.strip() in w for k in self.keywords)]
        context = [" ".join(words[max(0, idx-self.n):min(idx+self.n+1, len(words))]) for idx in found_index]

        return '|'.join(context)
    
    def __helper__(self, text):
        context = self.get_context(text).split('|')
        
        if len(context) > 0 and context[0] != "":
            all_scores = []
            for c in context:
                scores = self.get_sim(c)
                all_scores.append(self.label_from_max(scores))
            max_score = max(all_scores, key=itemgetter(2))
            if max_score[2] >= self.threshold:
                return max_score[0], max_score[1], max_score[2]
            else:
                return None, 0, None
        else:
            return None, 0, None
        
    def run(self, texts):
        df = pd.DataFrame(texts, columns=["text"])
        df['inferred_rule'], df['inferred_label'], df['inferred_confidence'] = zip(*df["text"].progress_apply(self.__helper__))
        return df

class ActivityMatch():
    def __init__(self, threshold=0.9):
        print("INIT", flush=True)
        if torch.cuda.is_available() == True:
            self.device = "cuda"
            self.batch_size = 2048
        else:
            self.device = "cpu"
            self.batch_size = 64

        self.threshold = threshold

        print("Preparing embeddings...", flush=True)
        self.embedding_model = SentenceTransformer("thenlper/gte-small", device=self.device)
        self.activities = pd.read_csv(impresources.files("JAAT.data") / "lexiconwex2023.csv").drop_duplicates()
        self.activities = self.activities.reset_index().drop("index", axis=1)
        self.act_embed = self.embedding_model.encode(self.activities.example.to_list(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.act_embed = self.act_embed.to(self.device)

        print("Setting up pipeline...", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/task-classifier-mini-improved2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "loyoladatamining/task-classifier-mini-improved2",
            use_fast=True,
            max_length=64,
            truncation=True
        )
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, max_length=64, device=self.device, truncation=True, batch_size=self.batch_size, num_workers=mp.cpu_count())
        print("Finished.", flush=True)

    def get_candidates(self, text):
        s = sent_tokenize(text.strip())
        all_data = [ss for ss in s if len(ss.split()) <= 48]

        positive = []
        predictions = []

        temp = ListDataset(all_data)
        for r in self.pipe(temp):
            predictions.append(r)

        count = 0
        for x, y in zip(all_data, predictions):
            if y['label'] == 'LABEL_1':
                positive.append(x)
                count += 1
        return positive
    
    def get_candidates_batch(self, texts):
        all_data = []
        for i, t in enumerate(texts):
            s = sent_tokenize(t.strip())
            all_data.extend([(i, ss) for ss in s if len(ss.split()) <= 48])

        positive = []
        predictions = []

        temp = ListDataset([x[1] for x in all_data])
        for r in self.pipe(temp):
            predictions.append(r)

        count = 0
        for x, y in zip(all_data, predictions):
            if y['label'] == 'LABEL_1':
                positive.append(x)
                count += 1
        return positive

    def get_activities(self, text):
        positive = self.get_candidates(text)
        if len(positive) == 0:
            return []

        q_embed = self.embedding_model.encode(positive, convert_to_tensor=True, batch_size=64)
        if self.device == "cuda":
            q_embed = q_embed.to(self.device)

        search = util.semantic_search(corpus_embeddings=self.act_embed, query_embeddings=q_embed, top_k=1)

        found = 0
        matched_acts = []
        for x in search:
            if x[0]["score"] >= self.threshold:
                found += 1
                matched_acts.append((str(self.activities.iloc[x[0]["corpus_id"]]["code"]), self.activities.iloc[x[0]["corpus_id"]]["activity"], self.activities.iloc[x[0]["corpus_id"]]["example"]))

        return matched_acts
    
    def get_activities_batch(self, texts):
        all_data = self.get_candidates_batch(texts)

        q_embed = self.embedding_model.encode([x[1] for x in all_data], convert_to_tensor=True, batch_size=64)
        if self.device == "cuda":
            q_embed = q_embed.to(self.device)

        search = util.semantic_search(corpus_embeddings=self.act_embed, query_embeddings=q_embed, top_k=1)

        found = 0
        matched_acts = []
        for _ in range(len(texts)):
            matched_acts.append([])
        for x, y in zip(search, all_data):
            if x[0]["score"] >= self.threshold:
                found += 1
                matched_acts[y[0]].append((str(self.activities.iloc[x[0]["corpus_id"]]["code"]), self.activities.iloc[x[0]["corpus_id"]]["activity"], self.activities.iloc[x[0]["corpus_id"]]["example"]))

        return matched_acts
    
class JobTag():
    def __init__(self, class_name, n=4):
        with open(impresources.files("JAAT.data") / "keywords.json", 'r') as f:
            self.keywords = json.load(f)

        self.classes = sorted([x for x in self.keywords])

        if class_name not in self.classes:
            print("Usage Error: please select one of the available classes:\n[{}]".format(" | ".join(self.classes)))
            return
        
        self.class_name = class_name
        filename = "{}.lzma".format(self.class_name)
        with open(impresources.files("JAAT.models") / filename, 'rb') as f:
            self.clf = compress_pickle.load(f, compression="lzma", set_default_extension=False)
        self.n = n

    def get_tag(self, text):
        contexts = get_context(text, keywords=self.keywords[self.class_name], n=self.n)
        if contexts is None:
            return (self.class_name, 0)
        res = self.clf.predict(contexts)
        return (self.class_name, 1 if sum(res) > 0 else 0)
    
    def get_tag_batch(self, texts, progress_bar=False):
        with mp.Pool(mp.cpu_count(), init_pool(self.clf)) as pool:
            c = pool.imap(partial(get_context, keywords=self.keywords[self.class_name], n=self.n), texts)
            all_contexts = list(c)
            if progress_bar == True:
                p = tqdm(pool.imap(classify, all_contexts), total=len(all_contexts))
            else:
                p = pool.imap(classify, all_contexts)
            res = list(p)
            pool.close()
        return res
    
class WageExtract():
    def __init__(self, batch_size=2048):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
            self.batch_size = batch_size
        else:
            self.device = "cpu"
            self.batch_size = 64

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

    def preproc(self, text):
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

    def get_largest_chunk(self, text, n=6):
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
    
    def extract_wage(self, predict):
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

    def get_wage(self, text):
        sentences = nltk.sent_tokenize(text)
        is_pay = self.ispay_pipe(sentences)
        if all(x["label"] == "LABEL_0" for x in is_pay):
            return "The provided text does not contain a wage statement."
        
        text = self.get_largest_chunk(self.preproc(text))
        pred = self.extract_wage(self.ner_pipe(text))
        freq = self.freq_pipe(text)[0]["label"]
        pred["frequency"] = freq
        return pred

    def get_wage_batch(self, texts):
        temp = []
        for i, x in enumerate(texts):
            temp.extend([(i, y) for y in nltk.sent_tokenize(x)])
        all_indices = [x[0] for x in temp]
        all_sentences = [x[1] for x in temp]

        batch = ListDataset(all_sentences)
        is_pay = []
        print("Predicting is_pay...", flush=True)
        for p in tqdm(self.ispay_pipe(batch), total=len(all_sentences)):
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
                indices.append(c)
                cands.append(texts[c])

        print("{} / {} contain wage statements.\n".format(len(indices), len(texts)), flush=True)

        new_batch = ListDataset(cands)

        print("Extracting wage...", flush=True)
        preds = []
        for p in tqdm(self.ner_pipe(new_batch), total=len(new_batch)):
            preds.append(self.extract_wage(p))
        
        print("\nPredicting pay frequency...", flush=True)
        freqs = []
        for p in tqdm(self.freq_pipe(new_batch), total=len(new_batch)):
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
        print("\nSummary: found {} wage statements.".format(total), flush=True)

        return ret