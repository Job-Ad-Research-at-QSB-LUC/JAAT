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

HF_TOKEN = None

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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
        s = nltk.sent_tokenize(text.strip())
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
            s = nltk.sent_tokenize(t.strip())
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

    def __init__(self):
        print("INIT", flush=True)
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"
        model = AutoModelForTokenClassification.from_pretrained("sjmeis/firmNER", token=HF_TOKEN, id2label={0: 'O', 1: 'B-ORG', 2: 'I-ORG'}, label2id={'O': 0, 'B-ORG': 1, 'I-ORG': 2})
        tokenizer = AutoTokenizer.from_pretrained("sjmeis/firmNER", token=HF_TOKEN)
        self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=self.device, aggregation_strategy="max")

        remove = string.punctuation
        remove = remove.replace(".", "").replace(",", "").replace("-", "").replace("&", "").replace("'","")
        self.pattern = r"[{}]".format(remove)

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