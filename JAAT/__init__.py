import os
import sys
import time
import threading
import re
from .config import JAAT_ART, SHOW_ART, VERSION, DESCRIPTION, CYAN, GREEN, RESET, BOLD, DIM

def get_visible_len(text: str) -> int:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', text))

class Loader:
    def __init__(self, desc="Loading JAAT..."):
        self.desc = desc
        self.done = False
        self.thread = None
        self.width = 64

    def _animate(self):
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while not self.done:
            sys.stdout.write(f"\r  {CYAN}{chars[idx % len(chars)]}{RESET} {self.desc}...")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)

    def __enter__(self):
        if SHOW_ART:
            top_bottom = "+" + "-" * (self.width - 2) + "+"
            side = f"{DIM}|{RESET}"
            
            print(f"\n{DIM}{top_bottom}{RESET}")
            
            for line in JAAT_ART.strip("\n").split("\n"):
                content = line.ljust(self.width - 4)
                print(f"{side} {CYAN}{content}{RESET} {side}")
            
            print(f"{side}" + " " * (self.width - 2) + f"{side}")
            
            welcome_msg = f"Welcome to JAAT v{VERSION}".center(self.width - 4)
            desc_msg = DESCRIPTION.center(self.width - 4)
            
            print(f"{side} {BOLD}{welcome_msg}{RESET} {side}")
            print(f"{side} {DIM}{desc_msg}{RESET} {side}")
            print(f"{DIM}{top_bottom}{RESET}\n")
            
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.done = True
        if self.thread:
            self.thread.join()
            sys.stdout.write(f"\r   {GREEN}✅ {self.desc}... Ready.{RESET}\n\n")
            sys.stdout.flush()

with Loader():
    os.environ["USE_TORCH"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    from .match import TaskMatch, TitleMatch, SkillMatch, ActivityMatch, AIMatch
    from .extract import FirmExtract, WageExtract, Readability
    from .label import CREAM, JobTag
    from .search import ConceptSearch
    from .utils import setup, chunker, validate_inputs, diagnostic, clear_cache, shutdown, toggle_progress

__all__ = [
    "setup", "chunker", "validate_inputs", "diagnostic", "clear_cache", "shutdown", "toggle_progress",
    "TaskMatch", "TitleMatch", "SkillMatch", "ActivityMatch", "AIMatch",
    "FirmExtract", "WageExtract", "Readability",
    "CREAM", "JobTag", "ConceptSearch"
]