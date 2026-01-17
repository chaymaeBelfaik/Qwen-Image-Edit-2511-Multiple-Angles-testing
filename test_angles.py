"""
Test script for multiple angle variations using SKS format
Includes comprehensive logging and metrics
"""
import os
import torch
import time
import datetime
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def create_angle_gif(image_paths, output_dir, duration=500):
    """
    Create an animated GIF from the generated angle variations.

    Args:
        image_paths: List of paths to generated images
        output_dir: Directory to save the GIF
        duration: Duration between frames in milliseconds
    """
    if not image_paths:
        print("No images to create GIF from")
        return

    # Load all images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            # Ensure consistent size (resize if needed)
            if img.size != (1024, 1024):
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")

    if not images:
        print("No valid images found for GIF creation")
        return

    # Create animated GIF
    gif_path = f"{output_dir}/angle_variations.gif"

    # Save as GIF with animation
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # Infinite loop
        optimize=True
    )

    print(f"GIF saved to: {gif_path}")
    print(f"Frames: {len(images)}, Duration: {duration}ms per frame")

# Try to import huggingface_hub for downloading LoRA
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Initialize pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_vram():
        return pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1024**3

    def get_gpu_info():
        """Get detailed GPU information"""
        try:
            gpu_name = pynvml.nvmlDeviceGetName(gpu_handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            total_mem = mem_info.total / 1024**3
            allocated_mem = mem_info.used / 1024**3
            free_mem = mem_info.free / 1024**3
            # Reserved is typically close to allocated in PyTorch
            reserved_mem = allocated_mem
            return {
                'name': gpu_name,
                'total': total_mem,
                'allocated': allocated_mem,
                'reserved': reserved_mem,
                'free': free_mem
            }
        except:
            return None

    METRICS_AVAILABLE = True
    print("✓ pynvml available for GPU monitoring")
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ pynvml not available. Install with: pip install pynvml")
    def get_vram():
        return 0.0
    def get_gpu_info():
        return None

def download_lora_if_needed(lora_path="qwen-image-edit-2511-multiple-angles-lora.safetensors", repo_id=None):
    """Download LoRA file if it doesn't exist"""
    if os.path.exists(lora_path):
        print(f"✓ LoRA file already exists: {lora_path}")
        return True

    if not HF_AVAILABLE:
        print("❌ Cannot download LoRA: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return False

    # Use provided repo_id or default to the correct one
    if repo_id is None:
        # Check environment variable first
        env_repo = os.getenv("QWEN_LORA_REPO")
        if env_repo:
            repo_id = env_repo
        else:
            # Use the correct repository provided by user
            repo_id = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"

    print(f"🔍 Using repository: {repo_id}")

    print(f"🔍 Checking repository: {repo_id}")

    # Verify repository exists
    try:
        from huggingface_hub import model_info
        info = model_info(repo_id)
        print(f"   ✅ Repository found: {repo_id}")
        print(f"   📦 Downloads: {getattr(info, 'downloads', 'N/A')}")
    except Exception as e:
        print(f"❌ Repository access failed: {e}")
        print("The repository may not exist, be private, or require authentication")
        print("\n💡 Troubleshooting:")
        print("1. Check if the repository exists on Hugging Face")
        print("2. If private, set HF_TOKEN environment variable")
        print("3. Or download manually and place in this directory")
        print(f"   Filename: {lora_path}")
        return False

    print(f"⬇️  Downloading LoRA from {repo_id}...")
    download_start = time.time()

    try:
        # Download the LoRA file
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=".",
            local_dir_use_symlinks=False,
            resume_download=True
        )

        # Find the safetensors file in the downloaded directory
        downloaded_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.safetensors') and 'multiple' in file.lower() and 'angle' in file.lower():
                    downloaded_files.append(os.path.join(root, file))

        # If no specific match, take any safetensors file
        if not downloaded_files:
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    if file.endswith('.safetensors'):
                        downloaded_files.append(os.path.join(root, file))

        if downloaded_files:
            # Move/rename the first safetensors file to our expected name
            source_file = downloaded_files[0]
            if source_file != lora_path:
                os.rename(source_file, lora_path)
                print(f"✓ Renamed {os.path.basename(source_file)} to {lora_path}")

            download_time = time.time() - download_start
            print(f"✓ Download completed in {download_time:.2f} seconds")
            print(f"✓ LoRA saved to: {os.path.abspath(lora_path)}")
            return True
        else:
            print("❌ No .safetensors file found in downloaded repository")
            return False

    except Exception as e:
        print(f"❌ Failed to download LoRA: {e}")
        print("The repository may not exist, be private, or require authentication")
        print("\n💡 To fix:")
        print("1. Set HF_TOKEN if repository is private:")
        print("   export HF_TOKEN='your-huggingface-token'")
        print("2. Or download manually from the repository")
        print(f"3. Place the file as: {lora_path}")
        return False

def load_pipeline_with_lora(lora_path="qwen-image-edit-2511-multiple-angles-lora.safetensors"):
    """Load pipeline with multiple-angles LoRA and track loading time"""
    print("Loading base model: Qwen/Qwen-Image-Edit-2511")

    # Clear GPU memory before loading
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Start loading timer
    loading_start_time = time.time()

    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16
        )
        pipeline.to('cuda')
        pipeline.set_progress_bar_config(disable=None)

        # Performance optimizations
        print("Enabling performance optimizations...")

        # TF32 Precision (faster on A100/Ampere+ GPUs)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ✓ TF32 precision enabled")

        # cuDNN Benchmark Mode (faster repeated operations)
        torch.backends.cudnn.benchmark = True
        print("  ✓ cuDNN benchmark enabled")
    except torch.cuda.OutOfMemoryError:
        print("❌ Out of memory during model loading. Try:")
        print("   1. Kill other Python processes: pkill -f python")
        print("   2. Clear GPU cache: python3 -c 'import torch; torch.cuda.empty_cache()'")
        print("   3. Restart your session")
        raise

    # Try to download LoRA if not present
    if not os.path.exists(lora_path):
        print(f"LoRA file not found: {lora_path}")
        print("Attempting to download from Hugging Face...")
        download_lora_if_needed(lora_path)

    # Load LoRA (required for this test)
    if not os.path.exists(lora_path):
        print(f"❌ LoRA file not found: {lora_path}")
        print("This test requires both base model and LoRA to function properly.")
        exit(1)

    try:
        print(f"Loading LoRA weights from {lora_path}...")
        pipeline.load_lora_weights(lora_path)
        print("LoRA loaded successfully!")
        lora_loaded = True
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        print("Cannot proceed without LoRA. Please check your PEFT installation.")
        exit(1)

    loading_time = time.time() - loading_start_time
    return pipeline, loading_time, lora_loaded

def generate_with_prompt(pipeline, image, prompt, output_filename, seed=42, log_file=None):
    """Generate image with given prompt and collect metrics"""
    print(f"\n🎬 Generating: {prompt}")
    if log_file:
        prompt_short = prompt.replace("<sks> ", "")
        log_file.write(f"🔬 ANALYZING: {prompt_short}\n")

    # Ensure image is in correct format for Qwen (needs list of images)
    if hasattr(image, 'mode') and image.mode != 'RGB':
        image = image.convert('RGB')
    input_image = [image]  # Qwen expects a list of images

    inputs = {
        "image": input_image,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 8,  # Reduced to 8 steps for faster generation
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    # Clear cache and synchronize
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Start metrics
    t_start = time.time()
    vram_start = get_vram() if METRICS_AVAILABLE else 0.0

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]

    # Synchronize and get final metrics
    torch.cuda.synchronize()
    generation_time = time.time() - t_start
    vram_peak = get_vram() if METRICS_AVAILABLE else 0.0

    # Save image
    output_image.save(output_filename)
    print(f"✅ Saved: {output_filename}")
    print(f"   ⏱️  Time: {generation_time:.2f}s")
    if METRICS_AVAILABLE:
        print(f"   🧠 Peak VRAM: {vram_peak:.2f} GB")

    return {
        "time": generation_time,
        "vram_peak": vram_peak,
        "prompt": prompt,
        "filename": output_filename,
        "image_size": output_image.size,
        "pixels": output_image.size[0] * output_image.size[1],
        "pixels_per_sec": (output_image.size[0] * output_image.size[1]) / generation_time if generation_time > 0 else 0,
        "steps": inputs["num_inference_steps"]
    }

# Test different SKS angle prompts for showing the girl from all angles
sks_prompts = [
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

def write_log_header(log_file, output_dir, gpu_info, initial_vram):
    """Write the log file header"""
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    log_file.write(f"Multiple Angles LoRA Test - {timestamp_str}\n")
    log_file.write("=" * 60 + "\n")
    log_file.write(f"Output directory: {os.path.abspath(output_dir)}\n")
    log_file.write(f"Log file: {os.path.abspath(log_file.name)}\n\n")

    if gpu_info:
        log_file.write(f"Using GPU: {gpu_info['name']}\n\n")

        log_file.write("=" * 60 + "\n")
        log_file.write("GPU Memory Info (Initial)\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"Total GPU Memory: {gpu_info['total']:.2f} GB\n")
        log_file.write(f"Allocated: {initial_vram:.2f} GB\n")
        reserved = initial_vram  # For consistency with PyTorch
        log_file.write(f"Reserved: {reserved:.2f} GB\n")
        log_file.write(f"Free: {gpu_info['free']:.2f} GB\n")
        log_file.write("=" * 60 + "\n\n")

    log_file.write(f"Testing all {len(sks_prompts)} angle variations:\n")
    for i, prompt in enumerate(sks_prompts):
        prompt_short = prompt.replace("<sks> ", "")
        log_file.write(f"  {i+1}: {prompt_short}\n")
    log_file.write("\n")

def write_model_loading_section(log_file, lora_path, loading_time, gpu_info):
    """Write the model loading section"""
    log_file.write("=" * 60 + "\n")
    log_file.write("LOADING MODEL WITH MULTIPLE ANGLES LORA\n")
    log_file.write("=" * 60 + "\n\n")

    log_file.write("Loading base transformer model...\n")
    log_file.write("Setting up scheduler...\n")
    log_file.write("Creating pipeline...\n")
    if os.path.exists(lora_path):
        log_file.write(f"Loading LoRA weights from {lora_path}...\n")
    log_file.write("Moving to device...\n\n")

    log_file.write(f"Model loading time: {loading_time:.2f} seconds\n\n")

    if gpu_info:
        current_gpu = get_gpu_info()
        if current_gpu:
            log_file.write("=" * 60 + "\n")
            log_file.write("GPU Memory Info (After Model Loading)\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Total GPU Memory: {current_gpu['total']:.2f} GB\n")
            log_file.write(f"Allocated: {current_gpu['allocated']:.2f} GB\n")
            reserved = current_gpu.get('reserved', current_gpu['allocated'])
            log_file.write(f"Reserved: {reserved:.2f} GB\n")
            log_file.write(f"Free: {current_gpu['free']:.2f} GB\n")
            log_file.write("=" * 60 + "\n\n")

            log_file.write(f"Peak GPU Memory Used (after loading): {current_gpu['allocated']:.2f} GB\n\n")
            
            # Show GPU name again for clarity
            if gpu_info:
                log_file.write(f"Using GPU: {gpu_info['name']}\n\n")

def write_benchmark_results(log_file, results):
    """Write the benchmark results section"""
    log_file.write("=" * 60 + "\n")
    # Get steps from first result
    steps_used = results[0].get('steps', 8) if results else 8
    log_file.write(f"MULTIPLE ANGLES MODEL - {steps_used} STEPS\n")
    log_file.write("=" * 60 + "\n\n")

    log_file.write("=" * 80 + "\n")
    log_file.write("BENCHMARK RESULTS\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"{'#':<4} | {'PROMPT':<50} | {'STEPS':<6} | {'TIME':<10} | {'VRAM':<8}\n")
    log_file.write("-" * 80 + "\n")

    for i, r in enumerate(results):
        prompt_short = r['prompt'].replace("<sks> ", "")[:48]
        steps = r.get('steps', 8)
        log_file.write(f"{i+1:<4} | {prompt_short:<50} | {steps:<6} | {r['time']:>8.2f}s | {r['vram_peak']:>6.2f} GB\n")

    log_file.write("\n")

def write_prompt_section(log_file, prompt_name="dior_angles"):
    """Write the prompt section before generation"""
    log_file.write("=" * 80 + "\n")
    log_file.write(f"PROMPT: {prompt_name}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Prompt preview: Testing multiple angle variations using SKS format...\n\n")

def write_final_summary(log_file, results, loading_time, output_dir, log_filename, gpu_info, lora_loaded):
    """Write the final summary section"""
    log_file.write("=" * 80 + "\n")
    log_file.write("FINAL SUMMARY\n")
    log_file.write("=" * 80 + "\n\n")

    log_file.write("Model: Qwen/Qwen-Image-Edit-2511\n")
    if lora_loaded:
        log_file.write("LoRA: qwen-image-edit-2511-multiple-angles-lora.safetensors\n")
    
    # Get steps from first result
    steps_used = results[0].get('steps', 8) if results else 8
    log_file.write(f"Inference Steps: {steps_used}\n")
    log_file.write("CFG Scale: 4.0\n")
    log_file.write(f"Model Loading Time: {loading_time:.2f} seconds\n")
    log_file.write(f"Total Angles Tested: {len(results)}\n")
    log_file.write(f"Output Directory: {os.path.abspath(output_dir)}\n")
    log_file.write(f"Log File: {os.path.abspath(log_filename)}\n\n")

    log_file.write("=" * 80 + "\n")
    log_file.write("STATISTICS (Successful Generations Only)\n")
    log_file.write("=" * 80 + "\n")

    successful_results = [r for r in results if r['time'] > 0]
    if successful_results:
        # Generation time (without model loading)
        total_generation_time = sum(r['time'] for r in successful_results)
        avg_generation_time = total_generation_time / len(successful_results)
        fastest_generation_time = min(r['time'] for r in successful_results)
        slowest_generation_time = max(r['time'] for r in successful_results)
        
        # Total time (with model loading)
        total_time_with_loading = loading_time + total_generation_time
        avg_time_with_loading = total_time_with_loading / len(successful_results)
        
        # Calculate throughput
        all_throughput = [r['pixels_per_sec'] / 1000000 for r in successful_results]
        avg_throughput = sum(all_throughput) / len(all_throughput)
        max_throughput = max(all_throughput)
        
        # Calculate time per step (use actual steps from first result)
        steps_per_gen = successful_results[0].get('steps', 8) if successful_results else 8
        all_step_times = [r['time'] / steps_per_gen * 1000 for r in successful_results]  # Convert to ms
        avg_step_time = sum(all_step_times) / len(all_step_times)
        fastest_step_time = min(all_step_times)
        
        # VRAM stats
        all_vram = [r['vram_peak'] for r in successful_results]
        avg_vram = sum(all_vram) / len(all_vram)
        max_vram = max(all_vram)
        min_vram = min(all_vram)

        log_file.write(f"Total Successful: {len(successful_results)}/{len(results)}\n")
        log_file.write(f"\nGeneration Time (without model loading):\n")
        log_file.write(f"  Average: {avg_generation_time:.2f}s\n")
        log_file.write(f"  Fastest: {fastest_generation_time:.2f}s\n")
        log_file.write(f"  Slowest: {slowest_generation_time:.2f}s\n")
        log_file.write(f"  Total: {total_generation_time:.2f}s\n")
        log_file.write(f"\nTotal Time (with model loading):\n")
        log_file.write(f"  Model Loading: {loading_time:.2f}s\n")
        log_file.write(f"  Generation: {total_generation_time:.2f}s\n")
        log_file.write(f"  Total: {total_time_with_loading:.2f}s\n")
        log_file.write(f"  Average per image (with loading): {avg_time_with_loading:.2f}s\n")
        log_file.write(f"Average Throughput: {avg_throughput:.2f}M px/sec\n")
        log_file.write(f"Max Throughput: {max_throughput:.2f}M px/sec\n")
        log_file.write(f"Average Time per Step: {avg_step_time:.1f}ms\n")
        log_file.write(f"Fastest Step Time: {fastest_step_time:.1f}ms\n")
        if METRICS_AVAILABLE:
            log_file.write(f"Average Peak VRAM: {avg_vram:.2f} GB\n")
            log_file.write(f"Max Peak VRAM: {max_vram:.2f} GB\n")
            log_file.write(f"Min Peak VRAM: {min_vram:.2f} GB\n")

    log_file.write("\n" + "=" * 80 + "\n\n")
    log_file.write("Test completed successfully!\n")

if __name__ == "__main__":
    # Setup timestamp and output directory
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    output_dir = f"mustang_angles_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Create log file
    log_filename = f"{output_dir}/test_log_{timestamp_str}.txt"
    log_file = open(log_filename, 'w')

    print("🎭 Multiple Angles LoRA Test - 1967 Mustang Fastback")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Log file: {os.path.abspath(log_filename)}")
    print()

    # Get initial GPU info
    gpu_info = get_gpu_info() if METRICS_AVAILABLE else None
    initial_vram = get_vram() if METRICS_AVAILABLE else 0.0

    if gpu_info:
        print(f"Using GPU: {gpu_info['name']}")
        print()
        print("=" * 60)
        print("GPU Memory Info (Initial)")
        print("=" * 60)
        print(f"Total GPU Memory: {gpu_info['total']:.2f} GB")
        print(f"Allocated: {initial_vram:.2f} GB")
        print(f"Free: {gpu_info['free']:.2f} GB")
        print("=" * 60)
        print()

    # Write log header
    write_log_header(log_file, output_dir, gpu_info, initial_vram)

    # Load pipeline and track loading time
    print("Loading model...")
    pipeline, loading_time, lora_loaded = load_pipeline_with_lora()

    # Write model loading section to log
    write_model_loading_section(log_file, "qwen-image-edit-2511-multiple-angles-lora.safetensors", loading_time, gpu_info)

    # Load the specific test image
    test_image_path = "1967 Mustang Fastback.jpg"
    if not os.path.exists(test_image_path):
        error_msg = f"❌ Test image not found: {test_image_path}"
        print(error_msg)
        log_file.write(error_msg + "\n")
        exit(1)

    image = Image.open(test_image_path)
    print(f"✅ Loaded test image: {test_image_path}")
    print(f"   Image size: {image.size}")
    print(f"   Image mode: {image.mode}")

    # Convert to RGB if necessary and handle transparency
    if image.mode in ('RGBA', 'LA', 'P'):
        print(f"   Converting from {image.mode} to RGB...")
        # Create white background for transparent images
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])  # Use alpha as mask
                image = background
            else:
                image = image.convert('RGB')
        else:
            image = image.convert('RGB')
    elif image.mode != 'RGB':
        print(f"   Converting from {image.mode} to RGB...")
        image = image.convert('RGB')

    # Resize if needed (Qwen might expect specific dimensions)
    target_size = (1024, 1024)  # Qwen expects 1024x1024
    if image.size != target_size:
        print(f"   Resizing from {image.size} to {target_size}...")
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    print(f"   Final image size: {image.size}, mode: {image.mode}")
    log_file.write(f"Loaded test image: {test_image_path} ({image.size[0]}x{image.size[1]}, mode: {image.mode})\n\n")

    # Collect results for final report
    results = []

    # Write prompt section to log
    write_prompt_section(log_file, "mustang_angles")

    # Generate images for each prompt
    print(f"\n🎬 Generating {len(sks_prompts)} angle variations...")
    print("=" * 80)

    for i, prompt in enumerate(sks_prompts):
        # Create filename from prompt (sanitize for filesystem)
        safe_name = prompt.replace("<sks> ", "").replace(" ", "_").replace("-", "_")
        filename = f"{output_dir}/angle_{i+1:02d}_{safe_name}.png"

        result = generate_with_prompt(pipeline, image, prompt, filename, seed=i+200, log_file=log_file)
        results.append(result)

        # Progress indicator
        progress = (i + 1) / len(sks_prompts) * 100
        print(f"   Progress: {progress:.1f}% ({i+1}/{len(sks_prompts)})")

    # Write benchmark results to log
    write_benchmark_results(log_file, results)

    # Write final summary to log
    write_final_summary(log_file, results, loading_time, output_dir, log_filename, gpu_info, lora_loaded)

    # Create animated GIF from all generated images
    print("\n🎬 Creating animated GIF...")
    try:
        create_angle_gif(generated_images, output_dir)
        print("✅ Animated GIF created successfully!")
    except Exception as e:
        print(f"⚠️  Failed to create GIF: {e}")

    # Close log file
    log_file.close()

    # Print final console summary
    print("\n" + "=" * 80)
    print("📊 MULTIPLE ANGLES TEST RESULTS")
    print("=" * 80)
    print(f"✅ Generation complete! Check the '{output_dir}' folder for {len(sks_prompts)} variations.")
    print(f"🎬 Animated GIF: {output_dir}/angle_variations.gif")
    print(f"📝 Detailed log saved to: {log_filename}")
    print("💡 Each variation shows the subject from a different camera angle using SKS format")
    print("=" * 80)
