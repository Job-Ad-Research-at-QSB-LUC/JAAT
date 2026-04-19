from typing import List, Optional, Any
from pathlib import Path
import ahocorasick
import pickle

from .base import logger
from .utils import progress_bar

class ConceptSearch():
    def __init__(self, concept_map: Optional[dict] = None, concept_file: Optional[str] = None) -> None:
        if concept_file == None:
            if concept_map is None:
                logger.error("Error: concept_map cannot be None!")
                return

            # map of {concept: code}
            self.concept_map = concept_map

            self.auto = ahocorasick.Automaton()
            for x in self.concept_map:
                self.auto.add_word(x.lower(), (x, self.concept_map[x]))
            self.auto.make_automaton()
        else:
            if Path(concept_file).is_file() == False:
                logger.error("Error: concept_file does not exist.")
                return
            
            with open(concept_file, 'rb') as f:
                self.auto = pickle.load(f)

    def get_concepts(self, text: str) -> List[Any]:
        codes = []
        text = text.lower()
        for idx, found in self.auto.iter(text):
            start = idx - len(found[0]) + 1
            start_check = True
            if start > 0:
                if text[start-1].isalnum() == True:
                    start_check = False

            end_check = True
            if idx < len(text) - 1:
                if text[idx+1].isalnum() == True:
                    end_check = False

            if start_check == True and end_check == True:
                codes.append(found[1])
        return codes

    def get_concepts_batch(self, texts: List[str]) -> List[List[Any]]:
        all_codes = []
        for text in progress_bar(texts, total=len(texts)):
            ret = set()
            for x in self.get_concepts(text):
                ret.add(x)
            all_codes.append(list(ret))

        return all_codes
