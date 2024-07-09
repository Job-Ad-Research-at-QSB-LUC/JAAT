import psutil
from transformers import pipeline
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import nltk
from tqdm.auto import tqdm
from pathlib import Path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import importlib_resources as impresources
import multiprocessing as mp

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
        all_data = []
        for i, t in enumerate(texts):
            positive = self.get_candidates(t)
            all_data.extend([(i, p) for p in positive])

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