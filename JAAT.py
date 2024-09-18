from transformers import pipeline
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import nltk
from tqdm.auto import tqdm
import pandas as pd
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
import pickle
from functools import partial

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
        with open(impresources.files("data") / "sub_dict.json", 'r') as f:
            self.sub_dict = json.load(f)

    def standardize(self, text):
        # account for boundaries
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
        tasks = pd.read_csv(impresources.files("data") / "Task_DWA.csv")[["Task ID", "Task"]].drop_duplicates()
        self.tasks = tasks.reset_index().drop("index", axis=1)
        self.task_embed = self.embedding_model.encode(tasks.Task.to_list(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.task_embed = self.task_embed.to(self.device)

        print("Setting up pipeline...", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(impresources.files("models") / "task-classifier-mini-improved2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            impresources.files("models") / "task-classifier-mini-improved2",
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
        #print("Found {} task sentences.".format(count), flush=True)
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
        #print("Found {} task sentences.".format(count), flush=True)
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

        #print("Matched {} tasks.".format(found), flush=True)
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

        #print("Matched {} tasks.".format(found), flush=True)
        return matched_tasks
    
class TitleMatch():

    def __init__(self):
        print("INIT", flush=True)
        if torch.cuda.is_available() == True:
            self.device = "cuda"
            self.batch_size = 2048
        else:
            self.device = "cpu"
            self.batch_size = 64

        print("Loading data...", flush=True)
        alt_titles = pd.read_csv(impresources.files("data") / "alternate_titles.csv")[["Alternate Title", "O*NET-SOC Code"]].drop_duplicates(["Alternate Title"]).dropna()
        titles = pd.read_csv(impresources.files("data") / "alternate_titles.csv")[["Title", "O*NET-SOC Code"]].drop_duplicates(["Title"]).dropna()
        alt_titles.columns = ["title", "code"]
        titles.columns = ["title", "code"]
        titles = pd.concat([titles, alt_titles])
        self.titles = titles.reset_index().drop("index", axis=1)

        print("Preparing embeddings...", flush=True)
        self.embedding_model = SentenceTransformer("thenlper/gte-small", device=self.device)
        self.title_embed = self.embedding_model.encode(self.titles.title.to_list(), convert_to_tensor=True, show_progress_bar=True)
        self.title_embed = self.title_embed.to(self.device)


    def get_title(self, text):
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, list):
            pass
        else:
            print("Error: input must be string or a list of strings.")
            return
        
        q_embed = self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=True)
        q_embed = q_embed.to(self.device)
        
        search = util.semantic_search(corpus_embeddings=self.title_embed, query_embeddings=q_embed, top_k=1)

        results = []
        for s in search:
            results.append((self.titles.title[s[0]["corpus_id"]], self.titles.code[s[0]["corpus_id"]], round(s[0]["score"], 3)))

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
        self.activities = pd.read_csv(impresources.files("data") / "lexiconwex2023.csv").drop_duplicates()
        self.activities = self.activities.reset_index().drop("index", axis=1)
        self.act_embed = self.embedding_model.encode(self.activities.example.to_list(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.act_embed = self.act_embed.to(self.device)

        print("Setting up pipeline...", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(impresources.files("models") / "task-classifier-mini-improved2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            impresources.files("models") / "task-classifier-mini-improved2",
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
        with open(impresources.files("data") / "keywords.json", 'r') as f:
            self.keywords = json.load(f)

        self.classes = sorted([x for x in self.keywords])

        if class_name not in self.classes:
            print("Usage Error: please select one of the available classes:\n[{}]".format(" | ".join(self.classes)))
            return
        
        self.class_name = class_name
        self.clf = pickle.load(open(impresources.files("models") / "jobtag" / self.class_name, 'rb'))
        self.n = n

    def get_tag(self, text):
        contexts = get_context(text)
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
