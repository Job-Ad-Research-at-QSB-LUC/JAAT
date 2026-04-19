import logging
import os

logger = logging.getLogger("JAAT")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

GLOBAL_SETTINGS = {
    "show_progress": True
}

MODEL_CACHE = {}

VERSION = "0.9.12 (beta)"
DESCRIPTION = "An NLP-powered toolkit for job ad text analysis."

CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

JAAT_ART = r"""
      ___           ___           ___           ___     
     /\  \         /\  \         /\  \         /\  \    
     \:\  \       /::\  \       /::\  \        \:\  \   
      \:\  \     /:/\:\  \     /:/\:\  \        \:\  \  
  ___  \:\  \   /::\~\:\  \   /::\~\:\  \       /::\  \ 
 /\  \  \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\     /:/\:\__\
 \:\  \ /:/  / \/__\:\/:/  / \/__\:\/:/  /    /:/  \/__/
  \:\  /:/  /       \::/  /       \::/  /    /:/  /     
   \:\/:/  /        /:/  /        /:/  /     \/__/      
    \::/  /        /:/  /        /:/  /                 
     \/__/         \/__/         \/__/                  
"""
SHOW_ART = os.isatty(1)