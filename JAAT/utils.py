import nltk
import torch
import importlib_resources as impresources
import sys
import platform
import gc

from .base import logger
from .matching import MODEL_CACHE

def setup():
    logger.info("--- JAAT Setup ---")

    # NLTK setup (one-time)
    nltk_requirements = {
        'tokenizers': ['punkt', 'punkt_tab'],
        'corpora': ['words']
    }
    for category, packages in nltk_requirements.items():
        for package in packages:
            try:
                nltk.data.find("{}/{}".format(category, package))
            except LookupError:
                logger.info("Downloading NLTK package: {}...".format(package))
                nltk.download(package, quiet=True)

    # necessary file check
    logger.info("Checking JAAT data files...")
    required_files = [
        "Task_DWA.csv",
        "lexiconwex2023.csv",
        "skills.csv",
        "ai_a6_5_redacted_final2.csv",
        "keywords.json",
        "sub_dict.json",
        "SOC_map.json",
        "title_embeddings.pickle"
    ]
    
    missing_files = []
    data_path = impresources.files("JAAT.data")
    for filename in required_files:
        if not (data_path / filename).is_file():
            missing_files.append(filename)
    
    if missing_files:
        logger.warning(f"❌ WARNING: The following data files are missing from JAAT/data/: {missing_files}")
        logger.warning("Please ensure these files are included in your package distribution.")
    else:
        logger.info("✅ All internal data files present.")

    # inform user of GPU availability 
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️ CUDA not detected. Running in CPU mode (larger tasks will be slower).")

    logger.info("\n--- Setup Complete ---")

def chunker(iterable, size):
    """
    Simple generator to break large datasets (lists of texts) into manageable chunks.
    """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def validate_inputs(texts):
    """
    Validator util to check batch texts before feeding into a JAAT function.
    """
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")
    return [str(t) if t is not None else "" for t in texts]

def diagnostic():
    print("--- JAAT Diagnostic Report ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")
        
    print("------------------------------")

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logger.info("VRAM and Memory cleared.")

def shutdown(clear_models=True):
    if clear_models:
        MODEL_CACHE.clear()
        logger.info("Global model cache cleared.")

    clear_cache()
    logger.info("JAAT resources fully released. Shutdown complete.")