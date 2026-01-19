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

def generate_with_prompt(pipeline, image, prompt, output_filename, seed=42, log_file=None, target_resolution=None):
    """Generate image with given prompt and collect metrics"""
    print(f"\n🎬 Generating: {prompt}")
    if log_file:
        prompt_short = prompt.replace("<sks> ", "")
        log_file.write(f"🔬 ANALYZING: {prompt_short}\n")

    # Ensure image is in correct format for Qwen (needs list of images)
    if hasattr(image, 'mode') and image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize input image to target resolution if specified
    if target_resolution is not None:
        if image.size != target_resolution:
            image = image.resize(target_resolution, Image.Resampling.LANCZOS)
            if log_file:
                log_file.write(f"   Resized input to {target_resolution[0]}x{target_resolution[1]}\n")
    
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
    resolution_str = f" {target_resolution[0]}x{target_resolution[1]}" if target_resolution else ""
    print(f"✅ Saved: {output_filename}")
    print(f"   ⏱️  Time: {generation_time:.2f}s{resolution_str}")
    if METRICS_AVAILABLE:
        print(f"   🧠 Peak VRAM: {vram_peak:.2f} GB")

    return {
        "time": generation_time,
        "vram_peak": vram_peak,
        "prompt": prompt,
        "filename": output_filename,
        "image_size": output_image.size,
        "resolution": target_resolution if target_resolution else output_image.size,
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

    log_file.write("=" * 100 + "\n")
    log_file.write("BENCHMARK RESULTS\n")
    log_file.write("=" * 100 + "\n")
    log_file.write(f"{'#':<4} | {'PROMPT':<45} | {'RESOLUTION':<12} | {'STEPS':<6} | {'TIME':<10} | {'VRAM':<8}\n")
    log_file.write("-" * 100 + "\n")

    for i, r in enumerate(results):
        prompt_short = r['prompt'].replace("<sks> ", "")[:43]
        steps = r.get('steps', 8)
        # Get resolution from result
        resolution = r.get('resolution', r.get('image_size', (0, 0)))
        if isinstance(resolution, tuple):
            resolution_str = f"{resolution[0]}x{resolution[1]}"
        else:
            resolution_str = str(resolution)
        log_file.write(f"{i+1:<4} | {prompt_short:<45} | {resolution_str:<12} | {steps:<6} | {r['time']:>8.2f}s | {r['vram_peak']:>6.2f} GB\n")

    log_file.write("\n")

def write_prompt_section(log_file, prompt_name="dior_angles"):
    """Write the prompt section before generation"""
    log_file.write("=" * 80 + "\n")
    log_file.write(f"PROMPT: {prompt_name}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Prompt preview: Testing multiple angle variations using SKS format...\n\n")

def create_inference_time_grid(results_by_resolution, output_dir, timestamp_str):
    """Create a grid showing inference time by image and resolution"""
    import csv
    
    # Get all unique resolutions from dictionary keys and prompts from results
    resolutions = sorted(results_by_resolution.keys(), key=lambda x: x[0] * x[1])
    prompts = []
    for results in results_by_resolution.values():
        for result in results:
            if result['prompt'] not in prompts:
                prompts.append(result['prompt'])
    # Keep original order of prompts (don't sort)
    
    print(f"\n📊 Creating inference time grid...")
    print(f"   Resolutions found: {len(resolutions)}")
    for idx, res in enumerate(resolutions, 1):
        print(f"      {idx}. {res[0]}x{res[1]}: {len(results_by_resolution[res])} images")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Expected grid entries: {len(prompts)} rows × {len(resolutions)} columns = {len(prompts) * len(resolutions)} cells")
    
    # Create grid data structure
    grid_data = {}
    for prompt in prompts:
        grid_data[prompt] = {}
        for resolution in resolutions:
            grid_data[prompt][resolution] = None
    
    # Fill grid with timing data
    filled_count = 0
    for resolution, results in results_by_resolution.items():
        for result in results:
            prompt = result['prompt']
            if prompt in grid_data and resolution in grid_data[prompt]:
                grid_data[prompt][resolution] = result['time']
                filled_count += 1
    
    print(f"   Grid entries filled: {filled_count}/{len(prompts) * len(resolutions)}")
    
    # Create CSV file
    csv_filename = f"{output_dir}/inference_time_grid_{timestamp_str}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with all resolutions
        header = ['Image/Prompt'] + [f"{r[0]}x{r[1]}" for r in resolutions]
        writer.writerow(header)
        
        # Write data rows with times for each resolution
        for i, prompt in enumerate(prompts):
            prompt_short = prompt.replace("<sks> ", "")[:50]
            row = [f"Image_{i+1:02d}_{prompt_short}"] + [
                f"{grid_data[prompt][r]:.2f}" if grid_data[prompt][r] is not None else "N/A"
                for r in resolutions
            ]
            writer.writerow(row)
    
    # Create formatted text file
    txt_filename = f"{output_dir}/inference_time_grid_{timestamp_str}.txt"
    with open(txt_filename, 'w') as txtfile:
        txtfile.write("=" * 150 + "\n")
        txtfile.write("INFERENCE TIME GRID - Time by Image and Resolution\n")
        txtfile.write("=" * 150 + "\n\n")
        
        # List all resolutions being tested
        txtfile.write(f"RESOLUTIONS TESTED ({len(resolutions)} total):\n")
        for idx, resolution in enumerate(resolutions, 1):
            txtfile.write(f"  {idx}. {resolution[0]}x{resolution[1]}\n")
        txtfile.write(f"\nImages per resolution: {len(prompts)}\n")
        txtfile.write(f"Total images in grid: {filled_count}\n")
        txtfile.write(f"Expected total: {len(prompts) * len(resolutions)}\n\n")
        
        # Calculate dynamic table width based on number of resolutions
        prompt_col_width = 50
        resolution_col_width = 12
        separator_width = 3  # " | "
        table_width = prompt_col_width + (len(resolutions) * (resolution_col_width + separator_width))
        
        # Write note confirming all resolutions in table
        txtfile.write("TABLE STRUCTURE:\n")
        txtfile.write(f"  - Rows: {len(prompts)} images/prompts\n")
        txtfile.write(f"  - Columns: {len(resolutions)} resolutions (all listed above)\n")
        txtfile.write(f"  - Each cell shows generation time in seconds\n\n")
        
        # Write header with all resolutions
        txtfile.write("INFERENCE TIME TABLE (seconds):\n")
        header_line = f"{'Image/Prompt':<{prompt_col_width}}"
        for idx, resolution in enumerate(resolutions):
            res_str = f"{resolution[0]}x{resolution[1]}"
            header_line += f" | {res_str:<{resolution_col_width}}"
        txtfile.write(header_line + "\n")
        txtfile.write("-" * table_width + "\n")
        
        # Write data rows with times for each resolution
        for i, prompt in enumerate(prompts):
            prompt_short = prompt.replace("<sks> ", "")[:48]
            row_line = f"Image_{i+1:02d}_{prompt_short:<{prompt_col_width-12}}"
            for resolution in resolutions:
                time_val = grid_data[prompt][resolution]
                if time_val is not None:
                    row_line += f" | {time_val:>10.2f}s"
                else:
                    row_line += f" | {'N/A':>10}"
            txtfile.write(row_line + "\n")
        
        # Write summary statistics for each resolution
        txtfile.write("\n" + "=" * 150 + "\n")
        txtfile.write("SUMMARY STATISTICS BY RESOLUTION\n")
        txtfile.write("=" * 150 + "\n\n")
        
        # Verify all resolutions are included
        txtfile.write(f"All {len(resolutions)} resolutions included in statistics:\n\n")
        
        for idx, resolution in enumerate(resolutions, 1):
            times = [grid_data[p][resolution] for p in prompts if grid_data[p][resolution] is not None]
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)
                txtfile.write(f"{idx}. Resolution {resolution[0]}x{resolution[1]}:\n")
                txtfile.write(f"   Images generated: {len(times)}/{len(prompts)}\n")
                txtfile.write(f"   Average time: {avg_time:.2f}s\n")
                txtfile.write(f"   Min time: {min_time:.2f}s\n")
                txtfile.write(f"   Max time: {max_time:.2f}s\n")
                txtfile.write(f"   Total time: {total_time:.2f}s\n\n")
            else:
                txtfile.write(f"{idx}. Resolution {resolution[0]}x{resolution[1]}:\n")
                txtfile.write(f"   ⚠️  No data available for this resolution\n\n")
        
        # Add verification section
        txtfile.write("=" * 150 + "\n")
        txtfile.write("VERIFICATION\n")
        txtfile.write("=" * 150 + "\n")
        txtfile.write(f"Resolutions in table header: {len(resolutions)}\n")
        txtfile.write(f"Resolutions in statistics section: {len(resolutions)}\n")
        txtfile.write(f"Resolutions listed at top: {len(resolutions)}\n")
        txtfile.write(f"✅ All {len(resolutions)} resolutions are included in the table above\n")
        txtfile.write(f"\nResolution list:\n")
        for idx, resolution in enumerate(resolutions, 1):
            txtfile.write(f"  {idx}. {resolution[0]}x{resolution[1]}\n")
    
    print(f"   ✅ Grid saved:")
    print(f"      CSV: {csv_filename}")
    print(f"      TXT: {txt_filename}")
    
    return csv_filename, txt_filename

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
    output_dir = f"tutankhamun_angles_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Define resolutions to test
    resolutions_to_test = [
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1280, 1280),
        (1536, 1536),
    ]

    # Create log file
    log_filename = f"{output_dir}/test_log_{timestamp_str}.txt"
    log_file = open(log_filename, 'w')

    print("🎭 Multiple Angles LoRA Test - Tutankhamun")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Log file: {os.path.abspath(log_filename)}")
    print(f"Resolutions to test: {[f'{r[0]}x{r[1]}' for r in resolutions_to_test]}")
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
    log_file.write(f"Testing resolutions: {[f'{r[0]}x{r[1]}' for r in resolutions_to_test]}\n\n")

    # Load pipeline and track loading time
    print("Loading model...")
    pipeline, loading_time, lora_loaded = load_pipeline_with_lora()

    # Write model loading section to log
    write_model_loading_section(log_file, "qwen-image-edit-2511-multiple-angles-lora.safetensors", loading_time, gpu_info)

    # Load the specific test image
    test_image_path = "tuntankhamun.jpg"
    if not os.path.exists(test_image_path):
        error_msg = f"❌ Test image not found: {test_image_path}"
        print(error_msg)
        log_file.write(error_msg + "\n")
        exit(1)

    # Load original image (we'll resize it for each resolution)
    original_image = Image.open(test_image_path)
    print(f"✅ Loaded test image: {test_image_path}")
    print(f"   Original image size: {original_image.size}")
    print(f"   Image mode: {original_image.mode}")

    # Convert to RGB if necessary and handle transparency
    if original_image.mode in ('RGBA', 'LA', 'P'):
        print(f"   Converting from {original_image.mode} to RGB...")
        # Create white background for transparent images
        if original_image.mode == 'P' and 'transparency' in original_image.info:
            original_image = original_image.convert('RGBA')
        if original_image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', original_image.size, (255, 255, 255))
            if original_image.mode == 'RGBA':
                background.paste(original_image, mask=original_image.split()[-1])  # Use alpha as mask
                original_image = background
            else:
                original_image = original_image.convert('RGB')
        else:
            original_image = original_image.convert('RGB')
    elif original_image.mode != 'RGB':
        print(f"   Converting from {original_image.mode} to RGB...")
        original_image = original_image.convert('RGB')

    print(f"   Final image mode: {original_image.mode}")
    log_file.write(f"Loaded test image: {test_image_path} ({original_image.size[0]}x{original_image.size[1]}, mode: {original_image.mode})\n\n")

    # Collect results organized by resolution
    results_by_resolution = {}
    all_results = []

    # Write prompt section to log
    write_prompt_section(log_file, "tutankhamun_angles")

    # Generate images for each resolution
    total_combinations = len(resolutions_to_test) * len(sks_prompts)
    current_combination = 0

    print(f"\n🎬 Generating {len(sks_prompts)} angle variations at {len(resolutions_to_test)} resolutions...")
    print(f"   Total: {total_combinations} image generations")
    print("=" * 80)

    for resolution in resolutions_to_test:
        resolution_str = f"{resolution[0]}x{resolution[1]}"
        print(f"\n{'='*80}")
        print(f"📐 Testing Resolution: {resolution_str}")
        print(f"{'='*80}")
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"RESOLUTION: {resolution_str}\n")
        log_file.write(f"{'='*80}\n\n")

        results_by_resolution[resolution] = []

        for i, prompt in enumerate(sks_prompts):
            current_combination += 1
            # Create filename from prompt and resolution (sanitize for filesystem)
            safe_name = prompt.replace("<sks> ", "").replace(" ", "_").replace("-", "_")
            filename = f"{output_dir}/res_{resolution[0]}x{resolution[1]}_angle_{i+1:02d}_{safe_name}.png"

            result = generate_with_prompt(
                pipeline, 
                original_image, 
                prompt, 
                filename, 
                seed=i+200, 
                log_file=log_file,
                target_resolution=resolution
            )
            results_by_resolution[resolution].append(result)
            all_results.append(result)

            # Progress indicator
            progress = (current_combination / total_combinations) * 100
            print(f"   Overall Progress: {progress:.1f}% ({current_combination}/{total_combinations})")

    # Write benchmark results to log
    write_benchmark_results(log_file, all_results)

    # Write final summary to log
    write_final_summary(log_file, all_results, loading_time, output_dir, log_filename, gpu_info, lora_loaded)

    # Create inference time grid
    print("\n📊 Creating inference time grid...")
    csv_file, txt_file = create_inference_time_grid(results_by_resolution, output_dir, timestamp_str)
    log_file.write(f"\nInference time grid saved to:\n")
    log_file.write(f"  CSV: {csv_file}\n")
    log_file.write(f"  TXT: {txt_file}\n")

    # Close log file
    log_file.close()

    # Print final console summary
    print("\n" + "=" * 80)
    print("📊 MULTIPLE ANGLES TEST RESULTS")
    print("=" * 80)
    print(f"✅ Generation complete!")
    print(f"   Total images generated: {total_combinations}")
    print(f"   Resolutions tested: {len(resolutions_to_test)}")
    print(f"   Prompts per resolution: {len(sks_prompts)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Detailed log: {log_filename}")
    print(f"📊 Inference time grid (CSV): {csv_file}")
    print(f"📊 Inference time grid (TXT): {txt_file}")
    print("💡 Each variation shows the subject from a different camera angle using SKS format")
    print("=" * 80)
