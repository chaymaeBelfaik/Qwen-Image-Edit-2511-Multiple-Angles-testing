"""
RunPod Handler for Qwen Image Edit Plus Pipeline with Multiple Angles LoRA Support
"""
import os
import sys
import logging
import tempfile
import base64
import traceback
import random
import string
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import torch
from PIL import Image
import io

# Try to import runpod, but allow testing without it
try:
    import runpod
    from runpod.serverless.utils import rp_upload, rp_cleanup
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    # Mock for testing
    class MockRunPod:
        class serverless:
            @staticmethod
            def start(config):
                pass
    runpod = MockRunPod()
    
    def rp_upload_upload_file(job_id, file_path):
        return file_path
    rp_upload = type('obj', (object,), {'upload_file': rp_upload_upload_file})()

from diffusers import QwenImageEditPlusPipeline

# Import download functions
try:
    from download_models import check_model_exists, download_all_models
    DOWNLOAD_AVAILABLE = True
except ImportError:
    DOWNLOAD_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_env_file(env_path: str = ".env"):
    """
    Load environment variables from .env file if it exists.
    This allows loading Wasabi credentials and other config from .env file.
    
    Args:
        env_path: Path to the .env file (default: ".env")
    """
    env_file = Path(env_path)
    if env_file.exists():
        logger.info(f"Loading environment variables from {env_path}")
        loaded_count = 0
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Set environment variable if not already set (env vars take precedence)
                    if key and value and key not in os.environ:
                        os.environ[key] = value
                        loaded_count += 1
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} environment variable(s) from {env_path}")
    # Silently skip if .env doesn't exist (not required)


# Load .env file before reading environment variables
load_env_file()

# Wasabi/S3 Configuration
WASABI_ACCESS_KEY = os.getenv("WASABI_ACCESS_KEY")
WASABI_SECRET_KEY = os.getenv("WASABI_SECRET_KEY")
WASABI_BUCKET = os.getenv("WASABI_BUCKET")
WASABI_ENDPOINT = os.getenv("WASABI_ENDPOINT")  # Optional - defaults to us-east-1 endpoint if not provided
WASABI_REGION = os.getenv("WASABI_REGION", "us-east-1")  # Default: us-east-1
USE_WASABI = os.getenv("USE_WASABI", "true").lower() == "true"  # Enable Wasabi by default
LORA_MODEL = os.getenv("LORA_MODEL", "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA")  # Multiple angles LoRA model
USE_LORA = os.getenv("USE_LORA", "true").lower() == "true"  # Enable LoRA by default
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen-Image-Edit-2511")  # Base model ID
AUTO_DOWNLOAD_MODELS = os.getenv("AUTO_DOWNLOAD_MODELS", "true").lower() == "true"  # Auto-download missing models

# Default inference steps for multiple angles LoRA (8 steps as per test_angles.py)
DEFAULT_INFERENCE_STEPS = 8

# Default SKS angle prompts
DEFAULT_SKS_PROMPTS = [
    # Front views
    "<sks> front view eye-level shot medium shot",
    "<sks> front view high-angle shot medium shot",
    "<sks> front view low-angle shot medium shot",
    "<sks> front view elevated shot medium shot",
    # Side views
    "<sks> right side view eye-level shot medium shot",
    "<sks> left side view eye-level shot medium shot",
    "<sks> right side view high-angle shot medium shot",
    "<sks> left side view low-angle shot medium shot",
    # Back views
    "<sks> back view eye-level shot medium shot",
    "<sks> back view high-angle shot medium shot",
    "<sks> back view low-angle shot medium shot",
    # Quarter views
    "<sks> front-left quarter view eye-level shot medium shot",
    "<sks> front-right quarter view eye-level shot medium shot",
    "<sks> back-left quarter view eye-level shot medium shot",
    "<sks> back-right quarter view eye-level shot medium shot",
    # Distance variations (front view)
    "<sks> front view eye-level shot close-up",
    "<sks> front view eye-level shot wide shot",
    "<sks> front view eye-level shot long shot",
]

# Global pipeline variable
pipeline = None

def generate_short_id(length: int = 8) -> str:
    """
    Generate a short random ID for file naming.
    
    Args:
        length: Length of the random ID (default: 8)
    
    Returns:
        Short random ID string (e.g., "id5dksi6")
    """
    chars = string.ascii_lowercase + string.digits
    random_id = ''.join(random.choice(chars) for _ in range(length))
    return f"id{random_id}"


def generate_timestamp() -> str:
    """
    Generate timestamp in format MMDDYYYYHHMMSS.
    
    Returns:
        Timestamp string (e.g., "12232025025400" for Dec 23, 2025, 02:54:00)
    """
    return datetime.now().strftime("%m%d%Y%H%M%S")


def generate_file_names() -> Tuple[str, str]:
    """
    Generate image and metadata file names.
    
    Returns:
        Tuple of (image_filename, metadata_filename)
    """
    timestamp = generate_timestamp()
    short_id = generate_short_id()
    image_filename = f"{timestamp}_{short_id}.png"
    metadata_filename = f"{timestamp}_{short_id}_metadata.json"
    return image_filename, metadata_filename


def generate_base_filename() -> str:
    """
    Generate a base filename (without extension) for pairing original and modified images.
    
    Returns:
        Base filename string (e.g., "12232025025400_id5dksi6")
    """
    timestamp = generate_timestamp()
    short_id = generate_short_id()
    return f"{timestamp}_{short_id}"


def upload_to_wasabi(file_path: str, metadata: Dict[str, Any] = None, 
                     image_filename: str = None, metadata_filename: str = None,
                     expiration_hours: int = 1) -> Optional[Dict[str, str]]:
    """
    Upload image file and metadata to Wasabi and return pre-signed URLs.
    
    Args:
        file_path: Path to the image file to upload
        metadata: Dictionary containing metadata to save as JSON
        image_filename: Custom image filename (if None, will be generated)
        metadata_filename: Custom metadata filename (if None, will be generated)
        expiration_hours: Number of hours the pre-signed URL should be valid (default: 1)
    
    Returns:
        Dictionary with 'image_url' and 'metadata_url', or None if upload fails
    """
    if not all([WASABI_ACCESS_KEY, WASABI_SECRET_KEY, WASABI_BUCKET]):
        logger.warning("Wasabi credentials not configured. Missing WASABI_ACCESS_KEY, WASABI_SECRET_KEY, or WASABI_BUCKET")
        return None
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Determine endpoint URL (defaults to us-east-1 if not provided)
        if WASABI_ENDPOINT:
            endpoint_url = WASABI_ENDPOINT
        else:
            # Default Wasabi endpoints by region
            region_endpoints = {
                "us-east-1": "https://s3.wasabisys.com",
                "us-east-2": "https://s3.us-east-2.wasabisys.com",
                "us-west-1": "https://s3.us-west-1.wasabisys.com",
                "eu-central-1": "https://s3.eu-central-1.wasabisys.com",
                "ap-northeast-1": "https://s3.ap-northeast-1.wasabisys.com",
            }
            endpoint_url = region_endpoints.get(WASABI_REGION, "https://s3.wasabisys.com")
            logger.info(f"Using Wasabi endpoint for region {WASABI_REGION}: {endpoint_url}")
        
        # Create S3 client for Wasabi
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=WASABI_ACCESS_KEY,
            aws_secret_access_key=WASABI_SECRET_KEY,
            region_name=WASABI_REGION
        )
        
        # Generate filenames if not provided
        if not image_filename or not metadata_filename:
            i_filename, m_filename = generate_file_names()
            image_filename = image_filename or i_filename
            metadata_filename = metadata_filename or m_filename
        
        # Determine storage path
        base_path = "ArtistV2_outputs/multiple_angles"
        
        # Generate S3 keys
        image_s3_key = f"{base_path}/{image_filename}"
        metadata_s3_key = f"{base_path}/{metadata_filename}"
        
        # Upload image file
        logger.info(f"Uploading image {file_path} to Wasabi bucket {WASABI_BUCKET} as {image_s3_key}")
        s3_client.upload_file(
            file_path,
            WASABI_BUCKET,
            image_s3_key,
            ExtraArgs={'ContentType': 'image/png'}
        )
        logger.info(f"Successfully uploaded image to Wasabi: {image_s3_key}")
        
        # Upload metadata file if provided
        metadata_url = None
        if metadata:
            # Create temporary metadata file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_metadata:
                json.dump(metadata, tmp_metadata, indent=2)
                tmp_metadata_path = tmp_metadata.name
            
            try:
                logger.info(f"Uploading metadata to Wasabi bucket {WASABI_BUCKET} as {metadata_s3_key}")
                s3_client.upload_file(
                    tmp_metadata_path,
                    WASABI_BUCKET,
                    metadata_s3_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                logger.info(f"Successfully uploaded metadata to Wasabi: {metadata_s3_key}")
                
                # Generate pre-signed URL for metadata
                expiration = timedelta(hours=expiration_hours)
                metadata_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': WASABI_BUCKET, 'Key': metadata_s3_key},
                    ExpiresIn=int(expiration.total_seconds())
                )
            finally:
                # Clean up temporary metadata file
                try:
                    os.unlink(tmp_metadata_path)
                except:
                    pass
        
        # Generate pre-signed URL for image (valid for expiration_hours)
        expiration = timedelta(hours=expiration_hours)
        image_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': WASABI_BUCKET, 'Key': image_s3_key},
            ExpiresIn=int(expiration.total_seconds())
        )
        
        logger.info(f"Generated pre-signed URLs valid for {expiration_hours} hour(s)")
        return {
            'image_url': image_url,
            'metadata_url': metadata_url,
            'image_filename': image_filename,
            'metadata_filename': metadata_filename
        }
        
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
        return None
    except ClientError as e:
        logger.error(f"Wasabi upload error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error uploading to Wasabi: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def upload_single_image_to_wasabi(file_path: str, s3_key: str, 
                                   content_type: str = 'image/png',
                                   expiration_hours: int = 1) -> Optional[str]:
    """
    Upload a single image file to Wasabi and return pre-signed URL.
    
    Args:
        file_path: Path to the image file to upload
        s3_key: S3 key (path) for the file in the bucket
        content_type: Content type of the file (default: 'image/png')
        expiration_hours: Number of hours the pre-signed URL should be valid (default: 1)
    
    Returns:
        Pre-signed URL string, or None if upload fails
    """
    if not all([WASABI_ACCESS_KEY, WASABI_SECRET_KEY, WASABI_BUCKET]):
        return None
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Determine endpoint URL
        if WASABI_ENDPOINT:
            endpoint_url = WASABI_ENDPOINT
        else:
            region_endpoints = {
                "us-east-1": "https://s3.wasabisys.com",
                "us-east-2": "https://s3.us-east-2.wasabisys.com",
                "us-west-1": "https://s3.us-west-1.wasabisys.com",
                "eu-central-1": "https://s3.eu-central-1.wasabisys.com",
                "ap-northeast-1": "https://s3.ap-northeast-1.wasabisys.com",
            }
            endpoint_url = region_endpoints.get(WASABI_REGION, "https://s3.wasabisys.com")
        
        # Create S3 client for Wasabi
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=WASABI_ACCESS_KEY,
            aws_secret_access_key=WASABI_SECRET_KEY,
            region_name=WASABI_REGION
        )
        
        # Upload image file
        logger.info(f"Uploading image {file_path} to Wasabi bucket {WASABI_BUCKET} as {s3_key}")
        s3_client.upload_file(
            file_path,
            WASABI_BUCKET,
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )
        logger.info(f"Successfully uploaded image to Wasabi: {s3_key}")
        
        # Generate pre-signed URL
        expiration = timedelta(hours=expiration_hours)
        image_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': WASABI_BUCKET, 'Key': s3_key},
            ExpiresIn=int(expiration.total_seconds())
        )
        
        return image_url
        
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
        return None
    except ClientError as e:
        logger.error(f"Wasabi upload error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error uploading to Wasabi: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def check_and_download_models():
    """
    Check if required models exist and download them if missing.
    
    Returns:
        True if models are available (either already existed or downloaded successfully), False otherwise
    """
    global LORA_MODEL  # Must be at the top of the function
    
    if not AUTO_DOWNLOAD_MODELS:
        logger.info("Auto-download is disabled. Skipping model check.")
        return True
    
    if not DOWNLOAD_AVAILABLE:
        logger.warning("Download module not available. Skipping model check.")
        return True
    
    try:
        # Check if base model exists
        base_model_exists = check_model_exists(BASE_MODEL)
        
        # Check if LoRA exists (if enabled)
        lora_exists = True
        if USE_LORA and LORA_MODEL:
            lora_exists = check_model_exists(LORA_MODEL)
        
        # If all models exist, skip download
        if base_model_exists and lora_exists:
            logger.info("All required models are available. Skipping download.")
            return True
        
        # Download missing models
        logger.info("Some models are missing. Starting download...")
        result = download_all_models(
            base_model_id=BASE_MODEL,
            lora_id=LORA_MODEL if USE_LORA and LORA_MODEL else None,
            use_lora=USE_LORA and bool(LORA_MODEL),
            cache_dir=None
        )
        
        # Handle both old (bool) and new (tuple) return types for backward compatibility
        if isinstance(result, tuple):
            success, lora_path = result
            if lora_path and USE_LORA:
                logger.info(f"LoRA downloaded to: {lora_path}")
                # Update LORA_MODEL to point to the downloaded file (ensures local file path for load_lora_weights)
                LORA_MODEL = lora_path
        else:
            # Old return type (bool) for backward compatibility
            success = result
        
        if success:
            logger.info("All models downloaded successfully.")
        else:
            logger.warning("Some models failed to download. Pipeline may fail to load.")
        
        return success
        
    except Exception as e:
        logger.error(f"Error checking/downloading models: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def initialize_pipeline():
    """Initialize the pipeline once at startup"""
    global pipeline
    if pipeline is None:
        # Check and download models if needed
        check_and_download_models()
        
        logger.info("Loading Qwen Image Edit Plus Pipeline...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.bfloat16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)
        logger.info(f"Pipeline moved to device: {device}")
        pipeline.set_progress_bar_config(disable=None)
        
        # Load LoRA if enabled
        if USE_LORA and LORA_MODEL:
            try:
                logger.info(f"Loading LoRA weights from {LORA_MODEL}...")
                pipeline.load_lora_weights(LORA_MODEL)
                logger.info(f"LoRA weights loaded successfully from {LORA_MODEL}")
                logger.info(f"Multiple Angles LoRA detected - using {DEFAULT_INFERENCE_STEPS} inference steps by default")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights from {LORA_MODEL}: {str(e)}")
                logger.warning("Continuing without LoRA weights...")
        
        logger.info("Pipeline loaded successfully")
    return pipeline

def decode_image(image_input):
    """Decode image from base64 string or URL"""
    if isinstance(image_input, str):
        # Check if it's base64
        if image_input.startswith('data:image'):
            # Data URL format: data:image/png;base64,...
            image_input = image_input.split(',')[1]
        try:
            # Try to decode as base64
            image_data = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_data))
            return image
        except:
            # If not base64, try to open as file path
            if os.path.exists(image_input):
                return Image.open(image_input)
            else:
                raise ValueError(f"Could not decode image: {image_input}")
    elif isinstance(image_input, dict):
        # Handle dict with 'image' key (base64)
        if 'image' in image_input:
            return decode_image(image_input['image'])
    return image_input

def create_angle_gif(images, output_path, duration=500, target_size=(1280, 1280)):
    """
    Create an animated GIF from the generated angle variations.
    
    Args:
        images: List of PIL Image objects
        output_path: Path to save the GIF
        duration: Duration between frames in milliseconds
        target_size: Target size for all frames (default: 1280x1280)
    
    Returns:
        Path to the created GIF file, or None if creation failed
    """
    if not images:
        logger.warning("No images to create GIF from")
        return None
    
    try:
        # Ensure all images are the same size
        processed_images = []
        for img in images:
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            processed_images.append(img)
        
        if not processed_images:
            logger.warning("No valid images found for GIF creation")
            return None
        
        # Create animated GIF
        processed_images[0].save(
            output_path,
            save_all=True,
            append_images=processed_images[1:],
            duration=duration,
            loop=0,  # Infinite loop
            optimize=True
        )
        
        logger.info(f"GIF created: {output_path} ({len(processed_images)} frames, {duration}ms per frame)")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create GIF: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_with_prompt(pipeline, image, prompt, seed=42, num_inference_steps=None, 
                         guidance_scale=1.0, true_cfg_scale=4.0, negative_prompt=" ", 
                         target_resolution=None):
    """Generate image with given prompt and collect metrics"""
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_INFERENCE_STEPS
    
    # Ensure image is in correct format for Qwen (needs list of images)
    if hasattr(image, 'mode') and image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize input image to target resolution if specified
    if target_resolution is not None:
        if image.size != target_resolution:
            image = image.resize(target_resolution, Image.Resampling.LANCZOS)
    
    input_image = [image]  # Qwen expects a list of images

    inputs = {
        "image": input_image,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    # Generate image (matching Lightning handler pattern)
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]

    # Ensure resolution is always a tuple (for use as dict key in grouping)
    res = target_resolution if target_resolution else output_image.size
    res = tuple(res) if not isinstance(res, tuple) else res  # Convert list -> tuple

    return {
        "image": output_image,
        "prompt": prompt,
        "resolution": res,
        "steps": num_inference_steps
    }

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for Qwen Image Edit Plus with Multiple Angles
    
    Expected input format:
    {
        "input": {
            "images": "base64_string_or_url" or ["base64_string_or_url", ...],
            "prompts": ["<sks> front view eye-level shot medium shot", ...],  # optional, defaults to all angles
            "resolutions": [[512, 512], [1024, 1024], ...],  # optional, defaults to [(1024, 1024)]
            "num_inference_steps": 8,  # optional, default 8
            "guidance_scale": 1.0,  # optional, default 1.0
            "true_cfg_scale": 4.0,  # optional, default 4.0
            "negative_prompt": " ",  # optional
            "seed": 0  # optional, default 0
        }
    }
    
    Images are uploaded to Wasabi (if configured) or RunPod storage, and pre-signed
    URLs (valid for 1 hour) are returned in the response. If Wasabi is not configured,
    the images are returned as base64 for backward compatibility.
    """
    try:
        # Initialize pipeline if not already done
        pipe = initialize_pipeline()
        
        # Get input data
        input_data = event.get("input", {})
        
        # Get images
        images_input = input_data.get("images", input_data.get("image"))
        if images_input is None:
            return {"error": "No images provided. Please provide 'images' or 'image' in input."}
        
        # Handle single image or list of images
        if isinstance(images_input, str):
            images = [decode_image(images_input)]
        elif isinstance(images_input, list):
            images = [decode_image(img) for img in images_input]
        else:
            images = [decode_image(images_input)]
        
        # Use first image as original
        original_image = images[0]
        
        # Get prompts (default to all SKS prompts if not provided)
        prompts = input_data.get("prompts", DEFAULT_SKS_PROMPTS)
        if not isinstance(prompts, list):
            prompts = [prompts]
        
        # Get resolutions (default to 1280x1280 if not provided)
        resolutions = input_data.get("resolutions", [(1280, 1280)])
        if not isinstance(resolutions, list):
            resolutions = [resolutions]
        # Convert to tuples if needed
        resolutions = [tuple(r) if isinstance(r, list) else r for r in resolutions]
        
        # Get optional parameters
        num_inference_steps = input_data.get("num_inference_steps", DEFAULT_INFERENCE_STEPS)
        guidance_scale = input_data.get("guidance_scale", 1.0)
        true_cfg_scale = input_data.get("true_cfg_scale", 4.0)
        negative_prompt = input_data.get("negative_prompt", " ")
        seed = input_data.get("seed", 0)
        
        # Track execution time
        execution_start_time = time.time()
        execution_start_datetime = datetime.now().isoformat()
        
        # Generate base filename for this batch
        base_filename = generate_base_filename()
        
        # Generate all angle variations
        logger.info(f"Generating {len(prompts)} angle variations at {len(resolutions)} resolution(s)...")
        generated_images = []
        all_results = []
        
        for res_idx, resolution in enumerate(resolutions):
            for prompt_idx, prompt in enumerate(prompts):
                # Generate unique seed for each image
                current_seed = seed + (res_idx * len(prompts) + prompt_idx)
                
                logger.info(f"Generating: {prompt} at {resolution[0]}x{resolution[1]}")
                
                # Track generation time for this image
                gen_start = time.time()
                result = generate_with_prompt(
                    pipe,
                    original_image,
                    prompt,
                    seed=current_seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    target_resolution=resolution
                )
                gen_time = time.time() - gen_start
                
                # Create filename
                safe_name = prompt.replace("<sks> ", "").replace(" ", "_").replace("-", "_")
                filename = f"{base_filename}_res_{resolution[0]}x{resolution[1]}_angle_{prompt_idx+1:02d}_{safe_name}.png"
                
                result["filename"] = filename
                result["prompt_index"] = prompt_idx
                result["resolution_index"] = res_idx
                result["time"] = gen_time
                generated_images.append(result)
                all_results.append({
                    "prompt": prompt,
                    "filename": filename,
                    "resolution": result["resolution"],
                    "generation_time": gen_time,
                    "steps": result["steps"]
                })
        
        execution_end_time = time.time()
        execution_end_datetime = datetime.now().isoformat()
        execution_duration = execution_end_time - execution_start_time
        
        # Save original image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_original:
            original_image.save(tmp_original.name, format="PNG")
            original_image_path = tmp_original.name
        original_filename = f"{base_filename}_original.png"
        
        # Save all generated images and upload them
        image_urls = []
        uploaded_files = []
        
        job_id = event.get("id", "unknown")
        base_path = "ArtistV2_outputs/multiple_angles"
        
        # Upload original image
        original_url = None
        if USE_WASABI:
            original_s3_key = f"{base_path}/{original_filename}"
            original_url = upload_single_image_to_wasabi(
                file_path=original_image_path,
                s3_key=original_s3_key,
                content_type='image/png',
                expiration_hours=1
            )
        else:
            if RUNPOD_AVAILABLE:
                original_url = rp_upload.upload_file(job_id=job_id, file_path=original_image_path)
        
        # Upload all generated images
        for gen_result in generated_images:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                gen_result["image"].save(tmp_file.name, format="PNG")
                tmp_path = tmp_file.name
                uploaded_files.append(tmp_path)
            
            image_url = None
            if USE_WASABI:
                s3_key = f"{base_path}/{gen_result['filename']}"
                image_url = upload_single_image_to_wasabi(
                    file_path=tmp_path,
                    s3_key=s3_key,
                    content_type='image/png',
                    expiration_hours=1
                )
            else:
                if RUNPOD_AVAILABLE:
                    image_url = rp_upload.upload_file(job_id=job_id, file_path=tmp_path)
                else:
                    # For testing without RunPod, return base64
                    buffered = io.BytesIO()
                    gen_result["image"].save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_url = f"data:image/png;base64,{img_base64}"
            
            image_urls.append({
                "url": image_url,
                "filename": gen_result["filename"],
                "prompt": gen_result["prompt"],
                "resolution": gen_result["resolution"],
                "generation_time": gen_result["time"]
            })
        
        # Create animated GIF(s) from all generated images (movie-like animation)
        # Group images by resolution to create separate GIFs per resolution
        gif_urls = []
        gif_filenames = []
        
        if len(generated_images) > 0:
            # Group images by resolution
            images_by_resolution = {}
            for gen_result in generated_images:
                res = gen_result["resolution"]
                # Ensure resolution is a tuple (for use as dict key)
                res = tuple(res) if not isinstance(res, tuple) else res
                if res not in images_by_resolution:
                    images_by_resolution[res] = []
                images_by_resolution[res].append(gen_result)
            
            # Create a GIF for each resolution
            for resolution, res_images in images_by_resolution.items():
                try:
                    # Extract images for this resolution
                    gif_images = [gen_result["image"] for gen_result in res_images]
                    
                    # Use the actual resolution chosen by the user
                    gif_resolution = resolution
                    
                    # Create temporary GIF file
                    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_gif:
                        gif_path = tmp_gif.name
                    
                    # Create the animated GIF using the user's chosen resolution
                    logger.info(f"Creating animated GIF with {len(gif_images)} frames at {gif_resolution[0]}x{gif_resolution[1]}...")
                    created_gif = create_angle_gif(
                        gif_images,
                        gif_path,
                        duration=500,  # 500ms per frame
                        target_size=gif_resolution
                    )
                    
                    if created_gif:
                        # Create filename with resolution info
                        if len(images_by_resolution) > 1:
                            gif_filename = f"{base_filename}_all_angles_animation_{gif_resolution[0]}x{gif_resolution[1]}.gif"
                        else:
                            gif_filename = f"{base_filename}_all_angles_animation.gif"
                        
                        # Upload GIF to Wasabi
                        gif_url = None
                        if USE_WASABI:
                            gif_s3_key = f"{base_path}/{gif_filename}"
                            gif_url = upload_single_image_to_wasabi(
                                file_path=gif_path,
                                s3_key=gif_s3_key,
                                content_type='image/gif',
                                expiration_hours=1
                            )
                        else:
                            if RUNPOD_AVAILABLE:
                                gif_url = rp_upload.upload_file(job_id=job_id, file_path=gif_path)
                        
                        # Clean up temporary GIF file
                        try:
                            os.unlink(gif_path)
                        except Exception as e:
                            logger.warning(f"Could not delete temporary GIF file {gif_path}: {e}")
                        
                        if gif_url:
                            gif_urls.append(gif_url)
                            gif_filenames.append(gif_filename)
                            logger.info(f"Animated GIF uploaded successfully: {gif_filename} ({gif_resolution[0]}x{gif_resolution[1]})")
                        else:
                            logger.warning(f"Failed to upload animated GIF for resolution {gif_resolution[0]}x{gif_resolution[1]}")
                except Exception as e:
                    logger.warning(f"Failed to create animated GIF for resolution {resolution}: {str(e)}")
                    logger.warning(traceback.format_exc())
        
        # For backward compatibility, use the first GIF URL if available
        gif_url = gif_urls[0] if gif_urls else None
        gif_filename = gif_filenames[0] if gif_filenames else None
        
        # Prepare metadata
        metadata = {
            "task": "multiple-angles-generation",
            "prompts": prompts,
            "resolutions": [list(r) for r in resolutions],
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "seed": seed,
            },
            "execution": {
                "start_time": execution_start_datetime,
                "end_time": execution_end_datetime,
                "duration_seconds": round(execution_duration, 2),
            },
            "model": {
                "name": BASE_MODEL,
                "lora": LORA_MODEL if USE_LORA else None,
            },
            "results": all_results,
            "files": {
                "original_filename": original_filename,
                "total_generated": len(generated_images),
            }
        }
        
        # Upload metadata
        metadata_url = None
        if USE_WASABI:
            try:
                import boto3
                from botocore.exceptions import ClientError
                
                # Create temporary metadata file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_metadata:
                    json.dump(metadata, tmp_metadata, indent=2)
                    tmp_metadata_path = tmp_metadata.name
                
                try:
                    if WASABI_ENDPOINT:
                        endpoint_url = WASABI_ENDPOINT
                    else:
                        region_endpoints = {
                            "us-east-1": "https://s3.wasabisys.com",
                            "us-east-2": "https://s3.us-east-2.wasabisys.com",
                            "us-west-1": "https://s3.us-west-1.wasabisys.com",
                            "eu-central-1": "https://s3.eu-central-1.wasabisys.com",
                            "ap-northeast-1": "https://s3.ap-northeast-1.wasabisys.com",
                        }
                        endpoint_url = region_endpoints.get(WASABI_REGION, "https://s3.wasabisys.com")
                    
                    s3_client = boto3.client(
                        's3',
                        endpoint_url=endpoint_url,
                        aws_access_key_id=WASABI_ACCESS_KEY,
                        aws_secret_access_key=WASABI_SECRET_KEY,
                        region_name=WASABI_REGION
                    )
                    
                    metadata_filename = f"{base_filename}_metadata.json"
                    metadata_s3_key = f"{base_path}/{metadata_filename}"
                    logger.info(f"Uploading metadata to Wasabi bucket {WASABI_BUCKET} as {metadata_s3_key}")
                    s3_client.upload_file(
                        tmp_metadata_path,
                        WASABI_BUCKET,
                        metadata_s3_key,
                        ExtraArgs={'ContentType': 'application/json'}
                    )
                    
                    # Generate pre-signed URL for metadata
                    expiration = timedelta(hours=1)
                    metadata_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': WASABI_BUCKET, 'Key': metadata_s3_key},
                        ExpiresIn=int(expiration.total_seconds())
                    )
                    logger.info(f"Successfully uploaded metadata to Wasabi: {metadata_s3_key}")
                finally:
                    try:
                        os.unlink(tmp_metadata_path)
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Failed to upload metadata: {str(e)}")
        
        # Clean up temporary files
        try:
            os.unlink(original_image_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file {original_image_path}: {e}")
        
        for tmp_path in uploaded_files:
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {e}")
        
        # Build response
        response = {
            "status": "success",
            "task": "multiple-angles-generation",
            "execution_time_seconds": round(execution_duration, 2),
            "total_images": len(generated_images),
            "prompts_count": len(prompts),
            "resolutions_count": len(resolutions),
        }
        
        if original_url:
            response["original_image_url"] = original_url
            response["original_filename"] = original_filename
        
        if metadata_url:
            response["metadata_url"] = metadata_url
        
        if gif_url:
            response["animation_gif_url"] = gif_url
            response["animation_gif_filename"] = gif_filename
            # If multiple resolutions, include all GIFs
            if len(gif_urls) > 1:
                response["animation_gifs"] = [
                    {"url": url, "filename": filename}
                    for url, filename in zip(gif_urls, gif_filenames)
                ]
        
        response["images"] = image_urls
        
        # Backward compatibility: image_url and image_filename
        # Use GIF URL if available, otherwise use first generated image URL
        if gif_url and gif_filename:
            response["image_url"] = gif_url
            response["image_filename"] = gif_filename
        elif len(image_urls) > 0:
            response["image_url"] = image_urls[0]["url"]
            response["image_filename"] = image_urls[0]["filename"]
        
        return response
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "status": "error",
            "traceback": traceback.format_exc() if RUNPOD_AVAILABLE else None
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    if RUNPOD_AVAILABLE:
        # Initialize RunPod serverless
        runpod.serverless.start({"handler": handler})
    else:
        print("RunPod not available. Use test_handler.py for testing.")
        print("To use with RunPod, install: pip install runpod")

