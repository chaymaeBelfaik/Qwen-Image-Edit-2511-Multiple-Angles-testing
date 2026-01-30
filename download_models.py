"""
Script to download required models (base model and LoRA) for Qwen Image Edit
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def check_model_exists(model_id: str, cache_dir: str = None) -> bool:
    """
    Check if a model exists in the cache.
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen-Image-Edit-2511")
        cache_dir: Optional cache directory (defaults to HF_HOME or ~/.cache/huggingface)
    
    Returns:
        True if model exists, False otherwise
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, token=None)
        if cache_dir:
            cache_path = Path(cache_dir) / "hub" / f"models--{model_id.replace('/', '--')}"
        else:
            # Default HuggingFace cache location
            hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            cache_path = Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}"
        
        # Check if the model directory exists and has files
        if cache_path.exists():
            # Check if it has the required files (at least config.json or similar)
            if any(cache_path.rglob("*.json")) or any(cache_path.rglob("*.safetensors")) or any(cache_path.rglob("*.bin")):
                logger.info(f"Model {model_id} found in cache at {cache_path}")
                return True
        
        logger.info(f"Model {model_id} not found in cache")
        return False
    except Exception as e:
        logger.warning(f"Error checking model {model_id}: {str(e)}")
        return False


def download_model(model_id: str, cache_dir: str = None, resume_download: bool = True) -> bool:
    """
    Download a model from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen-Image-Edit-2511")
        cache_dir: Optional cache directory
        resume_download: Whether to resume interrupted downloads
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading model: {model_id}")
        
        # Use snapshot_download to get the full model
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=resume_download,
            local_files_only=False
        )
        
        logger.info(f"Successfully downloaded model {model_id} to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {str(e)}")
        return False


def download_lora(lora_id: str, cache_dir: str = None, resume_download: bool = True) -> Optional[str]:
    """
    Download a LoRA adapter from HuggingFace and return the path to the safetensors file.
    
    Args:
        lora_id: HuggingFace LoRA model ID (e.g., "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA")
        cache_dir: Optional cache directory
        resume_download: Whether to resume interrupted downloads
    
    Returns:
        Path to the downloaded safetensors file, or None if download failed
    """
    try:
        logger.info(f"Downloading LoRA: {lora_id}")
        
        # LoRAs are typically stored as adapters, use snapshot_download
        local_dir = snapshot_download(
            repo_id=lora_id,
            cache_dir=cache_dir,
            resume_download=resume_download,
            local_files_only=False
        )
        
        # Find the safetensors file
        matches = list(Path(local_dir).rglob("*.safetensors"))
        if matches:
            lora_path = str(matches[0])
            logger.info(f"Successfully downloaded LoRA {lora_id} to {lora_path}")
            return lora_path
        else:
            logger.warning(f"Downloaded LoRA {lora_id} but no safetensors file found in {local_dir}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to download LoRA {lora_id}: {str(e)}")
        return None


def download_all_models(base_model_id: str = "Qwen/Qwen-Image-Edit-2511",
                        lora_id: str = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
                        use_lora: bool = True,
                        cache_dir: str = None) -> tuple[bool, Optional[str]]:
    """
    Download all required models (base model and LoRA if enabled).
    
    Args:
        base_model_id: Base model ID to download
        lora_id: LoRA model ID to download
        use_lora: Whether to download LoRA
        cache_dir: Optional cache directory
    
    Returns:
        Tuple of (success: bool, lora_path: Optional[str])
        success: True if all downloads successful, False otherwise
        lora_path: Path to downloaded LoRA safetensors file if downloaded, None otherwise
    """
    success = True
    lora_path = None
    
    # Download base model
    if not check_model_exists(base_model_id, cache_dir):
        logger.info(f"Base model {base_model_id} not found, downloading...")
        if not download_model(base_model_id, cache_dir):
            logger.error(f"Failed to download base model {base_model_id}")
            success = False
    else:
        logger.info(f"Base model {base_model_id} already exists, skipping download")
    
    # Download LoRA if enabled
    if use_lora and lora_id:
        if not check_model_exists(lora_id, cache_dir):
            logger.info(f"LoRA {lora_id} not found, downloading...")
            lora_path = download_lora(lora_id, cache_dir)
            if not lora_path:
                logger.error(f"Failed to download LoRA {lora_id}")
                success = False
        else:
            logger.info(f"LoRA {lora_id} already exists, skipping download")
            # Try to find existing file path
            if cache_dir:
                cache_path = Path(cache_dir) / "hub" / f"models--{lora_id.replace('/', '--')}"
            else:
                hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                cache_path = Path(hf_home) / "hub" / f"models--{lora_id.replace('/', '--')}"
            
            if cache_path.exists():
                matches = list(cache_path.rglob("*.safetensors"))
                if matches:
                    lora_path = str(matches[0])
    
    return success, lora_path


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Qwen Image Edit")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model ID (default: from BASE_MODEL env var or Qwen/Qwen-Image-Edit-2511)")
    parser.add_argument("--lora", type=str, default=None,
                       help="LoRA model ID (default: from LORA_MODEL env var or fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA)")
    parser.add_argument("--no-lora", action="store_true",
                       help="Skip LoRA download")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Cache directory for models")
    
    args = parser.parse_args()
    
    # Use environment variables if command-line args not provided
    base_model_id = args.base_model or os.getenv("BASE_MODEL", "Qwen/Qwen-Image-Edit-2511")
    lora_id = args.lora or os.getenv("LORA_MODEL", "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA")
    
    result = download_all_models(
        base_model_id=base_model_id,
        lora_id=lora_id,
        use_lora=not args.no_lora,
        cache_dir=args.cache_dir
    )
    
    # Handle both old (bool) and new (tuple) return types for backward compatibility
    if isinstance(result, tuple):
        success, lora_path = result
        if success:
            logger.info("All models downloaded successfully!")
            if lora_path:
                logger.info(f"LoRA file path: {lora_path}")
            sys.exit(0)
        else:
            logger.error("Some models failed to download")
            sys.exit(1)
    else:
        # Old return type (bool) for backward compatibility
        if result:
            logger.info("All models downloaded successfully!")
            sys.exit(0)
        else:
            logger.error("Some models failed to download")
            sys.exit(1)


if __name__ == "__main__":
    main()

