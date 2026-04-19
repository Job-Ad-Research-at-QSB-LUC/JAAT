from typing import List, Tuple, Union, Generator, Any
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import importlib_resources as impresources
import pickle
import json

from .base import ListDataset, get_device_settings, sent_tokenize
from .config import MODEL_CACHE, logger
from .utils import progress_bar

def get_shared_model(model_name: str, device: str) -> SentenceTransformer:
    if model_name not in MODEL_CACHE:
        model = SentenceTransformer(model_name, device=device)
        # if device == "cuda":
        #     model = model.half()
        MODEL_CACHE[model_name] = model
    return MODEL_CACHE[model_name]

class TaskMatch():
    def __init__(self, threshold: float = 0.87, embedding_model: str = "thenlper/gte-small", classification_model: str = "loyoladatamining/task-classifier-mini-v3") -> None:
        logger.info("Initalizing TaskMatch...")
        self.device, self.batch_size = get_device_settings()

        self.threshold = threshold

        logger.info("Preparing embeddings...")
        self.embedding_model = get_shared_model(embedding_model, self.device)
        tasks = pd.read_csv(impresources.files("JAAT.data") / "Task_DWA.csv")[["Task ID", "Task"]].drop_duplicates()
        self.tasks = tasks.reset_index().drop("index", axis=1)
        self.task_embed = self.embedding_model.encode(tasks.Task.to_list(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.task_embed = self.task_embed.to(self.device)
        self.task_embed = torch.nn.functional.normalize(self.task_embed, p=2, dim=1)

        logger.info("Setting up pipeline...")
        self.model = AutoModelForSequenceClassification.from_pretrained(classification_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            classification_model,
            use_fast=True,
            max_length=64,
            truncation=True
        )
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, max_length=64, device=self.device, truncation=True, batch_size=self.batch_size, num_workers=mp.cpu_count())
        logger.info("Finished.")

    def get_candidates(self, text: str) -> List[str]:
        s = sent_tokenize(text.strip())
        all_data = [ss for ss in s if len(ss.split()) <= 48 and len(ss.split()) > 4]

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
    
    def get_candidates_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        all_data = []
        for i, t in enumerate(texts):
            s = sent_tokenize(t.strip())
            all_data.extend([(i, ss) for ss in s if len(ss.split()) <= 48 and len(ss.split()) > 4])

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

    def get_tasks(self, text: str) -> List[Tuple[str, str]]:
        positive = self.get_candidates(text)
        if len(positive) == 0:
            return []

        q_embed = self.embedding_model.encode(positive, convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.task_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_tasks = []
        for score, idx in zip(max_scores, max_indices):
            if score >= self.threshold:
                task_row = self.tasks.iloc[idx]
                matched_tasks.append((str(task_row["Task ID"]), task_row["Task"]))

        return matched_tasks
    
    def get_tasks_batch(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        all_data = self.get_candidates_batch(texts)
        if len(all_data) == 0:
            return [[] for _ in range(len(texts))]

        q_embed = self.embedding_model.encode([x[1] for x in all_data], convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.task_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_tasks = []
        for _ in range(len(texts)):
            matched_tasks.append([])

        for i, (score, idx) in enumerate(zip(max_scores, max_indices)):
            if score >= self.threshold:
                text_idx = all_data[i][0]
                task_row = self.tasks.iloc[idx]
                matched_tasks[text_idx].append((str(task_row["Task ID"]), task_row["Task"]))

        return matched_tasks
    
class TitleMatch():
    ## Note: since the title embeddings are pre-computed here, using the non-default embedding model will not work as intended!
    def __init__(self, batch_size: int = 16, embedding_model: str = "thenlper/gte-small") -> None:
        logger.info("Initializing TitleMatch...")
        self.device, _ = get_device_settings()

        logger.info("Loading data...")
        with open(impresources.files("JAAT.data") / "SOC_map.json", 'r') as f:
            self.codes = json.load(f)
        with open(impresources.files("JAAT.data") / "title_embeddings.pickle", 'rb') as pkl:
            embed = pickle.load(pkl)
        self.title_embed = embed.to(self.device)
        self.title_embed = torch.nn.functional.normalize(self.title_embed, p=2, dim=1)

        logger.info("Preparing embeddings...")
        self.embedding_model = get_shared_model(embedding_model, self.device)

        logger.info("Loading title models...")
        self.value_model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/title_value", num_labels=1, problem_type="regression")
        self.value_tokenizer = AutoTokenizer.from_pretrained("loyoladatamining/title_value")
        self.value_pipe = pipeline("text-classification", model=self.value_model, tokenizer=self.value_tokenizer, device=self.device, function_to_apply="none")

        self.feature_model = AutoModelForSequenceClassification.from_pretrained("loyoladatamining/title_feature").to(self.device)
        self.feature_tokenizer = AutoTokenizer.from_pretrained("loyoladatamining/title_feature")
        self.feature_batch = batch_size

        logger.info("Finished.")

    def batch(self, iterable: List[Any], n: int = 1) -> Generator[List[Any], None, None]:
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def sigmoid(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        return 1 / (1 + np.exp(-x))

    @torch.no_grad()
    def get_title(self, text: Union[str, List[str]]) -> List[Tuple[str, float, float, str]]:
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, list):
            pass
        else:
            logger.error("Error: input must be string or a list of strings.")
            return
        
        logger.info("Extracting title values...")
        values = []
        temp = ListDataset(text)
        for res in progress_bar(self.value_pipe(temp, batch_size=self.feature_batch), total=len(text)):
            values.append(round(res["score"], 1))

        logger.info("Extracting title features...")
        features = []
        batches = self.batch(text, n=self.feature_batch)
        batches = ListDataset(list(batches))
        for b in progress_bar(batches, total=len(batches)):
            inputs = self.feature_tokenizer.batch_encode_plus(b, truncation=True, max_length=32, padding="max_length", return_tensors="pt").to(self.device)
            outputs = self.feature_model(inputs.input_ids)
            logits = outputs.logits
            with torch.no_grad():
                for l in logits:
                    res = (self.sigmoid(l.cpu()) > 0.98).numpy().astype(int).reshape(-1)
                    temp = sorted([self.feature_model.config.id2label[x] for x in np.where(res == 1)[0]])
                    if "none" in temp and len(temp) > 1:
                        temp = [x for x in temp if x != "none"]
                    features.append(";".join(temp))
  
        logger.info("Matching titles to codes...")
        q_embed = self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=True)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)
        
        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.title_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        results = []
        for i, (score, idx) in enumerate(zip(max_scores, max_indices)):
            results.append((self.codes[idx], score, values[i], features[i]))

        return results
    
class ActivityMatch(TaskMatch):
    def __init__(self, threshold: float = 0.9, embedding_model: str = "thenlper/gte-small") -> None:
        super().__init__(threshold=threshold, embedding_model=embedding_model)
        
        self.activities = pd.read_csv(impresources.files("JAAT.data") / "lexiconwex2023.csv").drop_duplicates().reset_index(drop=True)
        self.act_embed = self.embedding_model.encode(self.activities.example.to_list(), convert_to_tensor=True)
        self.act_embed = self.act_embed.to(self.device)
        self.act_embed = torch.nn.functional.normalize(self.act_embed, p=2, dim=1)

    def get_activities(self, text: str) -> List[Tuple[str, str, str]]:
        positive = self.get_candidates(text)
        if len(positive) == 0:
            return []

        q_embed = self.embedding_model.encode(positive, convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.act_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_scores = max_scores.cpu().tolist()
        max_indices = max_indices.cpu().tolist()

        matched_acts = []
        for score, idx in zip(max_scores, max_indices):
            if score >= self.threshold:
                act_row = self.activities.iloc[idx]
                matched_acts.append((str(act_row["code"]), act_row["activity"], act_row["example"]))

        return matched_acts
    
    def get_activities_batch(self, texts: List[str]) -> List[List[Tuple[str, str, str]]]:
        all_data = self.get_candidates_batch(texts)
        if len(all_data) == 0:
            return [[] for _ in range(len(texts))]

        q_embed = self.embedding_model.encode([x[1] for x in all_data], convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.act_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_acts = []
        for _ in range(len(texts)):
            matched_acts.append([])

        for i, (score, idx) in enumerate(zip(max_scores, max_indices)):
            if score >= self.threshold:
                text_idx = all_data[i][0]
                act_row = self.activities.iloc[idx]
                matched_acts[text_idx].append((str(act_row["code"]), act_row["activity"], act_row["example"]))

        return matched_acts
    
class SkillMatch():
    def __init__(self, threshold: float = 0.87, embedding_model: str = "thenlper/gte-large", classification_model: str = "loyoladatamining/skill-classifier-base-v2") -> None:
        logger.info("Initializing SkillMatch...")
        self.device, self.batch_size = get_device_settings()

        self.threshold = threshold

        logger.info("Preparing embeddings...")
        self.embedding_model = get_shared_model(embedding_model, self.device)
        self.skills_df = pd.read_csv(impresources.files("JAAT.data") / "skills.csv")
        self.skills = list(set(self.skills_df["label"]))
        self.skill_embed = self.embedding_model.encode(self.skills, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.skill_embed = self.skill_embed.to(self.device)
        self.skill_embed = torch.nn.functional.normalize(self.skill_embed, p=2, dim=1)

        self.skill_map = dict(zip(self.skills_df.label, self.skills_df["EuropaCode"]))

        logger.info("Setting up pipeline...")
        self.model = AutoModelForSequenceClassification.from_pretrained(classification_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            classification_model,
            use_fast=True,
            max_length=64,
            truncation=True
        )
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, max_length=64, device=self.device, truncation=True, batch_size=self.batch_size, num_workers=mp.cpu_count())
        logger.info("Finished.")

    def get_candidates(self, text: str) -> List[str]:
        s = sent_tokenize(text.strip())
        all_data = [ss for ss in s if len(ss.split()) <= 48 and len(ss.split()) > 4]

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
    
    def get_candidates_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        all_data = []
        for i, t in enumerate(texts):
            s = sent_tokenize(t.strip())
            all_data.extend([(i, ss) for ss in s if len(ss.split()) <= 48 and len(ss.split()) > 4])

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

    def get_skills(self, text: str) -> List[Tuple[str, str]]:
        positive = self.get_candidates(text)
        if len(positive) == 0:
            return []

        q_embed = self.embedding_model.encode(positive, convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.skill_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_skills = []
        for score, idx in zip(max_scores, max_indices):
            if score >= self.threshold:
                skill_row = self.skills[idx]
                matched_skills.append((skill_row, self.skill_map[skill_row]))

        return matched_skills
    
    def get_skills_batch(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        all_data = self.get_candidates_batch(texts)
        if not all_data:
            return [[] for _ in range(len(texts))]

        q_embed = self.embedding_model.encode([x[1] for x in all_data], convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.skill_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_skills = []
        for _ in range(len(texts)):
            matched_skills.append([])

        for i, (score, idx) in enumerate(zip(max_scores, max_indices)):
            if score >= self.threshold:
                text_idx = all_data[i][0]
                skill_row = self.skills[idx]
                matched_skills[text_idx].append((skill_row, self.skill_map[skill_row]))
        return matched_skills
    
class AIMatch():
    def __init__(self, threshold: float = 0.87, embedding_model: str = "thenlper/gte-small", classification_model: str = "loyoladatamining/ai-classifier-small-v4") -> None:
        logger.info("Initializing AIMatch...")
        if torch.cuda.is_available() == True:
            self.device = "cuda"
            self.batch_size = 2048
        else:
            self.device = "cpu"
            self.batch_size = 64

        self.threshold = threshold

        logger.info("Preparing embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        self.ai_df = pd.read_csv(impresources.files("JAAT.data") / "ai_a6_5_redacted_final2.csv")
        self.ai = self.ai_df["Statement"].to_list()
        self.ai_embed = self.embedding_model.encode(self.ai, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
        self.ai_embed = self.ai_embed.to(self.device)
        self.ai_embed = torch.nn.functional.normalize(self.ai_embed, p=2, dim=1)

        self.ai_map = dict(zip(self.ai_df["Statement"], self.ai_df["Code"]))
        self.score_map = dict(zip(self.ai_df["Statement"], self.ai_df["Score"]))

        logger.info("Setting up pipeline...")
        self.model = AutoModelForSequenceClassification.from_pretrained(classification_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            classification_model,
            use_fast=True,
            max_length=128,
            truncation=True
        )
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, max_length=128, device=self.device, truncation=True, batch_size=self.batch_size, num_workers=8)
        logger.info("Finished.")

    def get_candidates(self, text: str) -> List[str]:
        s = sent_tokenize(text.strip())
        all_data = [ss for ss in s if len(ss.split()) <= 64 and len(ss.split()) >= 4]

        positive = []
        predictions = []

        temp = ListDataset(all_data)
        for r in self.pipe(temp):
            predictions.append(r)

        count = 0
        for x, y in zip(all_data, predictions):
            if y['label'] == 'LABEL_1':
                positive.append((x, y["score"]))
                count += 1
        return positive
    
    def get_candidates_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        all_data = []
        for i, t in enumerate(texts):
            s = sent_tokenize(t.strip())
            all_data.extend([(i, ss) for ss in s if len(ss.split()) <= 64 and len(ss.split()) >= 4])

        positive = []
        predictions = []

        temp = ListDataset([x[1] for x in all_data])
        for r in self.pipe(temp):
            predictions.append(r)

        count = 0
        for x, y in zip(all_data, predictions):
            if y['label'] == 'LABEL_1':
                positive.append((x, y["score"]))
                count += 1
        return positive

    def get_ai(self, text: str) -> Tuple[List[Tuple[str, str]], int, float, List[float], List[float]]:
        positive = self.get_candidates(text)
        if len(positive) == 0:
            return [], 0, 0.0, [], []

        q_embed = self.embedding_model.encode([x[0] for x in positive], convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.ai_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_ai = []
        total_score = 0
        count = 0
        binary_scores = []
        match_scores = []
        for i, (score, idx) in enumerate(zip(max_scores, max_indices)):
            if score >= self.threshold:
                ai_row = self.ai[idx]
                matched_ai.append((ai_row, self.ai_map[ai_row]))
                match_scores.append(score)
                total_score += self.score_map[ai_row]
                count += 1
                binary_scores.append(round(positive[i][1], 3))

        return matched_ai, count, round(total_score / len(matched_ai), 3) if len(matched_ai) > 0 else 0, binary_scores, match_scores
    
    def get_ai_batch(self, texts: List[str]) -> List[Tuple[List[Tuple[str, str]], int, float, List[float], List[float]]]:
        all_data = self.get_candidates_batch(texts)
        if len(all_data) == 0: 
            return [[] for _ in range(len(texts))], [0]*len(texts), [0.0]*len(texts), [[] for _ in range(len(texts))], [[] for _ in range(len(texts))]

        q_embed = self.embedding_model.encode([x[0][1] for x in all_data], convert_to_tensor=True, batch_size=64)
        q_embed = q_embed.to(self.device)
        q_embed = torch.nn.functional.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_scores = torch.mm(q_embed, self.ai_embed.T)
        max_scores, max_indices = torch.max(sim_scores, dim=1)

        max_scores = [round(x, 3) for x in max_scores.cpu().tolist()]
        max_indices = max_indices.cpu().tolist()

        matched_ai = []
        scores = []
        counts = []
        binary_scores = []
        match_scores = []
        for _ in range(len(texts)):
            matched_ai.append([])
            scores.append(0)
            counts.append(0)
            binary_scores.append([])
            match_scores.append([])

        for i, (score, idx) in enumerate(zip(max_scores, max_indices)):
            if score >= self.threshold:
                text_idx = all_data[i][0][0]
                ai_row = self.ai[idx]
                matched_ai[text_idx].append((ai_row, self.ai_map[ai_row]))
                match_scores[text_idx].append(score)
                scores[text_idx] += self.score_map[ai_row]
                counts[text_idx] += 1
                binary_scores[text_idx].append(round(all_data[i][1], 3))

        return matched_ai, counts, [round(s / len(x), 3) if len(x) > 0 else 0 for s, x in zip(scores, matched_ai)], binary_scores, match_scores