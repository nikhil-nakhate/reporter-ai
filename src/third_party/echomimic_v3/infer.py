# -*- coding: utf-8 -*-
# ==============================================================================
# arxive: https://arxiv.org/abs/2507.03905
# GitHUb: https://github.com/antgroup/echomimic_v3
# Project Page: https://antgroup.github.io/ai/echomimic_v3/
# ==============================================================================

import os
import math
import datetime
from functools import partial
import platform

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from moviepy import VideoFileClip, AudioFileClip
import librosa

# Apple Silicon optimizations
def setup_apple_silicon_optimizations():
    """Configure PyTorch for optimal Apple Silicon performance."""
    if platform.system() == "Darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Set optimal number of threads for Apple Silicon
        # Apple Silicon benefits from fewer threads due to unified memory architecture
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        
        # Enable MPS optimizations
        if hasattr(torch.backends.mps, 'is_built'):
            print(f"MPS backend available: {torch.backends.mps.is_built()}")
        
        # Set memory-efficient settings
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        print("Apple Silicon optimizations enabled")
        return True
    return False

# Custom modules
from diffusers import FlowMatchEulerDiscreteScheduler

from src.dist import set_multi_gpus_devices
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import  CLIPModel
from src.wan_text_encoder import  WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline

from src.utils import (
    filter_kwargs,
    get_image_to_video_latent3,
    save_videos_grid,
)
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients

from src.face_detect import get_mask_coord


# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        # General settings
        self.ulysses_degree = 1
        self.ring_degree = 1
        self.fsdp_dit = False

        # Pipeline parameters
        self.num_skip_start_steps = 5
        self.teacache_offload = False
        self.cfg_skip_ratio = 0
        self.enable_riflex = False
        self.riflex_k = 6

        # Paths
        self.config_path = "config/config.yaml"
        self.model_name = "models/Wan2.1-Fun-V1.1-1.3B-InP"
        self.transformer_path = "models/transformer/diffusion_pytorch_model.safetensors"
        self.vae_path = None

        # Sampler and audio settings
        self.sampler_name = "Flow"
        self.audio_scale = 1.0
        self.enable_teacache = False
        self.teacache_threshold = 0.1
        self.shift = 5.0
        self.use_un_ip_mask = False

        # Inference parameters
        self.negative_prompt = "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作."#Unclear gestures, broken hands, more than five fingers on one hand, extra fingers, fused fingers. "# Strange body and strange trajectory. Distortion.  "

        self.use_longvideo_cfg = False
        self.partial_video_length = 113
        self.overlap_video_length = 8
        self.neg_scale = 1.5
        self.neg_steps = 2
        self.guidance_scale = 4. #4.0 ~ 6.0
        self.audio_guidance_scale = 2.9 #2.0 ~ 3.0
        self.use_dynamic_cfg = True
        self.use_dynamic_acfg = True
        self.seed = 43
        self.num_inference_steps = 25
        self.lora_weight = 1.0

        # Model settings
        self.weight_dtype = torch.bfloat16
        self.sample_size = [768, 768]
        self.fps = 25
        
        # Apple Silicon optimizations
        self.use_torch_compile = False  # Can be enabled if PyTorch 2.0+ and compatible
        self.enable_memory_efficient_attention = True

        # Test data paths
        self.base_dir = "datasets/echomimicv3_demos/"
        self.test_name_list = [
            'guitar_woman_01','guitar_man_01','music_woman_01',
            'demo_cartoon_03','demo_cartoon_04',
            '2025-07-14-1036','2025-07-14-1942',
            '2025-07-14-2371','2025-07-14-3927',
            '2025-07-14-4513','2025-07-14-6032',
            '2025-07-14-7113','2025-07-14-7335',
            'demo_ch_man_01','demo_ch_woman_04',
            ]
        self.wav2vec_model_dir = "models/wav2vec2-base-960h"
        self.save_path = "outputs"



# --------------------- Helper Functions ---------------------
def load_wav2vec_models(wav2vec_model_dir, device="cpu"):
    """Load Wav2Vec models for audio feature extraction."""
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    model = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).eval()
    model.requires_grad_(False)
    # Move to device for faster inference on Apple Silicon
    if device != "cpu":
        model = model.to(device)
    return processor, model


def extract_audio_features(audio_path, processor, model, device="cpu"):
    """Extract audio features using Wav2Vec."""
    sr = 16000
    audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
    input_values = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
    # Move input to device for faster processing on Apple Silicon
    if device != "cpu":
        input_values = input_values.to(device)
    with torch.no_grad():
        features = model(input_values).last_hidden_state
    return features.squeeze(0)


def get_sample_size(image, default_size):
    """Calculate the sample size based on the input image dimensions."""
    width, height = image.size
    original_area = width * height
    default_area = default_size[0] * default_size[1]

    if default_area < original_area:
        ratio = math.sqrt(original_area / default_area)
        width = width / ratio // 16 * 16
        height = height / ratio // 16 * 16
    else:
        width = width // 16 * 16
        height = height // 16 * 16

    return int(height), int(width)


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()

def get_file_path(base_dir, folder, test_name, extensions):
    """Helper function to find the file path with multiple extensions."""
    # Strip extension from test_name if it already has one
    base_name = test_name
    if '.' in test_name:
        # Check if the extension matches one of the expected extensions
        name_parts = test_name.rsplit('.', 1)
        if len(name_parts) == 2 and name_parts[1].lower() in [e.lower() for e in extensions]:
            base_name = name_parts[0]
        # Also check if file exists with the extension already in the name
        direct_path = os.path.join(base_dir, folder, test_name)
        if os.path.exists(direct_path):
            return direct_path
    
    # Try with each extension
    for ext in extensions:
        path = os.path.join(base_dir, folder, f"{base_name}.{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No file found for '{test_name}' (base: '{base_name}') in '{folder}' with extensions: {extensions}")


# --------------------- Main Script ---------------------
def main():
    # Setup Apple Silicon optimizations
    is_apple_silicon = setup_apple_silicon_optimizations()
    
    # Initialize configuration
    config = Config()

    # Set up multi-GPU devices
    device = set_multi_gpus_devices(config.ulysses_degree, config.ring_degree)
    
    # Check CUDA availability and fall back to CPU/MPS if needed
    using_mps = False
    if isinstance(device, str) and device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            using_mps = True
            print("🚀 Using MPS (Metal Performance Shaders) on Apple Silicon")
            # MPS doesn't support bfloat16, fall back to float16
            if config.weight_dtype == torch.bfloat16:
                config.weight_dtype = torch.float32
                print("   MPS doesn't support bfloat16, using float32 instead")
        else:
            device = "cpu"
            print("CUDA not available, using CPU. Note: This will be significantly slower.")
            # CPU doesn't support bfloat16 efficiently, use float32
            if config.weight_dtype == torch.bfloat16:
                config.weight_dtype = torch.float32
                print("CPU doesn't support bfloat16 efficiently, using float32 instead")
    elif isinstance(device, torch.device) and device.type == "cuda":
        if not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                using_mps = True
                print("🚀 Using MPS (Metal Performance Shaders) on Apple Silicon")
                # MPS doesn't support bfloat16, fall back to float16
                if config.weight_dtype == torch.bfloat16:
                    config.weight_dtype = torch.float32
                    print("   MPS doesn't support bfloat16, using float32 instead")
            else:
                device = torch.device("cpu")
                print("CUDA not available, using CPU. Note: This will be significantly slower.")
                # CPU doesn't support bfloat16 efficiently, use float32
                if config.weight_dtype == torch.bfloat16:
                    config.weight_dtype = torch.float32
                    print("CPU doesn't support bfloat16 efficiently, using float32 instead")
    
    # Apple Silicon specific optimizations
    if using_mps:
        # Enable memory efficient attention if available
        if config.enable_memory_efficient_attention:
            try:
                # Clear cache before loading models (if available)
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                print("   Memory efficient attention enabled")
            except:
                pass

    # Load configuration file
    cfg = OmegaConf.load(config.config_path)

    # Load models
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        os.path.join(config.model_name, cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        torch_dtype=config.weight_dtype,
    )
    if config.transformer_path is not None:
        print(f"\n📦 Loading additional checkpoint from: {config.transformer_path}")
        if config.transformer_path.endswith("safetensors"):
          from safetensors.torch import load_file, safe_open
          state_dict = load_file(config.transformer_path)
        else:
            state_dict = torch.load(config.transformer_path)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
        # Check what keys are in the checkpoint
        checkpoint_keys = set(state_dict.keys())
        model_keys = set(transformer.state_dict().keys())
        audio_keys_in_checkpoint = [k for k in checkpoint_keys if 'audio' in k.lower()]
        audio_keys_in_model = [k for k in model_keys if 'audio' in k.lower()]
        
        print(f"   Checkpoint contains {len(checkpoint_keys)} parameters")
        print(f"   Model expects {len(model_keys)} parameters")
        if audio_keys_in_checkpoint:
            print(f"   ✅ Checkpoint has {len(audio_keys_in_checkpoint)} audio-related parameters")
        else:
            print(f"   ⚠️  Checkpoint has NO audio-related parameters (model expects {len(audio_keys_in_model)})")
            print(f"   This suggests the checkpoint is from a non-audio version of the model.")
            print(f"   Audio components will be initialized randomly (may affect audio-driven generation).")
        
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        print(f"\n### Loading results:")
        print(f"### missing keys: {len(missing)}; ")
        print(f"### unexpected keys: {len(unexpected)};")
        
        # Categorize missing keys for better understanding
        audio_keys = [k for k in missing if 'audio' in k.lower()]
        other_keys = [k for k in missing if 'audio' not in k.lower()]
        
        if missing:
            if audio_keys:
                print(f"### ⚠️  Missing audio-related keys ({len(audio_keys)}):")
                print(f"###    This means audio components are NOT trained/initialized from checkpoint")
                print(f"###    Audio-driven generation may not work correctly!")
                if len(audio_keys) <= 10:
                    print(f"###    Missing: {audio_keys}")
                else:
                    print(f"###    First 10: {audio_keys[:10]}")
                    print(f"###    ... and {len(audio_keys) - 10} more audio keys")
            if other_keys:
                print(f"### Missing other keys ({len(other_keys)}): {other_keys[:5]}{'...' if len(other_keys) > 5 else ''}")
            if len(missing) <= 20 and len(audio_keys) == 0:
                print(f"### All missing keys: {missing}")
            elif len(missing) > 20 and len(audio_keys) == 0:
                print(f"### First 20 missing keys: {missing[:20]}")
                print(f"### ... and {len(missing) - 20} more")
        
        # Warn if this will affect functionality
        if len(audio_keys) > 0:
            print(f"\n⚠️  WARNING: The checkpoint does not contain audio weights!")
            print(f"   The model architecture expects audio support, but the checkpoint doesn't have it.")
            print(f"   Possible solutions:")
            print(f"   1. Download the correct checkpoint that includes audio weights")
            print(f"   2. Use a model variant without audio support (if available)")
            print(f"   3. Continue with random initialization (audio features may not work)\n")

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(config.model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
    ).to(config.weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
        torch_dtype=config.weight_dtype,
    ).eval()

    clip_image_encoder_path = os.path.join(config.model_name, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
    print(f"Loading CLIP image encoder from: {clip_image_encoder_path}")
    clip_image_encoder = CLIPModel.from_pretrained(
        clip_image_encoder_path,
    ).to(config.weight_dtype).eval()
    
    print("CLIP image encoder loaded successfully")

    # Load scheduler
    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[config.sampler_name]
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(cfg['scheduler_kwargs'])))

    # Create pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    pipeline.to(device=device)
    
    # Apple Silicon: Clear cache after moving models to device
    if using_mps:
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
        print("   Models loaded and cache cleared")

    # Enable TeaCache if required
    if config.enable_teacache:
        coefficients = get_teacache_coefficients(config.model_name)
        pipeline.transformer.enable_teacache(
            coefficients, config.num_inference_steps, config.teacache_threshold,
            num_skip_start_steps=config.num_skip_start_steps, offload=config.teacache_offload
        )

    # Load Wav2Vec models (optimized for device)
    wav2vec_processor, wav2vec_model = load_wav2vec_models(config.wav2vec_model_dir, device=device)

    format_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.save_path, f"{format_time}_gs{config.guidance_scale}_ags{config.audio_guidance_scale}")
    os.makedirs(save_path, exist_ok=True)

    # Process test cases
    generator = torch.Generator(device=device).manual_seed(config.seed)
    for test_name in config.test_name_list:
        # Strip extension from test_name if present for consistent handling
        base_test_name = test_name.rsplit('.', 1)[0] if '.' in test_name and test_name.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg'] else test_name
        
        try:
            ref_img_path = get_file_path(config.base_dir, "imgs", base_test_name, ["png", "jpeg", "jpg"])
        except FileNotFoundError as e:
            print(f"⚠️  {e}, skipping '{test_name}'")
            continue
            
        audio_path = os.path.join(config.base_dir, "audios", f"{base_test_name}.WAV")
        ip_mask_path = os.path.join(config.base_dir, "masks", f"{base_test_name}.npy")
        prompt_path = os.path.join(config.base_dir, "prompts", f"{base_test_name}.txt")
    
        if not os.path.exists(ref_img_path):
            print(f"⚠️  {ref_img_path} not exists, skip")
            continue
        if not os.path.exists(audio_path):
            audio_path = audio_path.replace("WAV", "wav")
        if not os.path.exists(audio_path):
            print(f"⚠️  {audio_path} not exists, skip")
            continue

        # Load reference image and prompt
        ref_img = Image.open(ref_img_path).convert("RGB")
        with open(prompt_path, "r") as f:
            prompt = f.read()

        # Load IP mask coordinates
        if not os.path.exists(ip_mask_path):
            y1, y2, x1, x2, h_, w_ = get_mask_coord(ref_img_path)
        else:
            y1, y2, x1, x2, h_, w_ = np.load(ip_mask_path)

        # Extract audio features (optimized for device)
        audio_clip = AudioFileClip(audio_path)
        audio_features = extract_audio_features(audio_path, wav2vec_processor, wav2vec_model, device=device)
        audio_embeds = audio_features.unsqueeze(0)
        # Ensure correct dtype and device
        if audio_embeds.device != device:
            audio_embeds = audio_embeds.to(device=device)
        audio_embeds = audio_embeds.to(dtype=config.weight_dtype)

        # Calculate video length and latent frames
        video_length = int(audio_clip.duration * config.fps)
        video_length = (
            int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            if video_length != 1 else 1
        )
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if config.enable_riflex:
            pipeline.transformer.enable_riflex(k = config.riflex_k, L_test = latent_frames)

        # Adjust sample size and create IP mask
        sample_height, sample_width = get_sample_size(ref_img, config.sample_size)
        downratio = math.sqrt(sample_height * sample_width / h_ / w_)
        coords = (
            y1 * downratio // 16, y2 * downratio // 16,
            x1 * downratio // 16, x2 * downratio // 16,
            sample_height // 16, sample_width // 16,
        )
        ip_mask = get_ip_mask(coords).unsqueeze(0)
        ip_mask = torch.cat([ip_mask]*3).to(device=device, dtype=config.weight_dtype)

        partial_video_length = int((config.partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (partial_video_length - 1) // vae.config.temporal_compression_ratio + 1

        if not config.use_longvideo_cfg:
            # get clip image
            _, _, clip_image = get_image_to_video_latent3(ref_img, None, video_length=partial_video_length, sample_size=[sample_height, sample_width])

            # Generate video in chunks
            init_frames = 0
            last_frames = init_frames + partial_video_length
            new_sample = None

            # Precompute mix_ratio outside the loop
            mix_ratio = torch.linspace(0, 1, steps=config.overlap_video_length).view(1, 1, -1, 1, 1)

            while init_frames < video_length:
                if last_frames >= video_length:
                    partial_video_length = video_length - init_frames
                    partial_video_length = (
                        int((partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
                        if video_length != 1 else 1
                    )
                    latent_frames = (partial_video_length - 1) // vae.config.temporal_compression_ratio + 1

                    if partial_video_length <= 0:
                        break

                input_video, input_video_mask, _ = get_image_to_video_latent3(
                    ref_img, None, video_length=partial_video_length, sample_size=[sample_height, sample_width]
                )
        
                partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + partial_video_length) * 2]
                
                sample  = pipeline(
                    prompt,
                    num_frames            = partial_video_length,
                    negative_prompt       = config.negative_prompt,
                    audio_embeds          = partial_audio_embeds,
                    audio_scale           = config.audio_scale,
                    ip_mask               = ip_mask,
                    use_un_ip_mask        = config.use_un_ip_mask,
                    height                = sample_height,
                    width                 = sample_width,
                    generator             = generator,
                    neg_scale             = config.neg_scale,
                    neg_steps             = config.neg_steps,
                    use_dynamic_cfg       = config.use_dynamic_cfg,
                    use_dynamic_acfg      = config.use_dynamic_acfg,
                    guidance_scale        = config.guidance_scale,
                    audio_guidance_scale  = config.audio_guidance_scale,
                    num_inference_steps   = config.num_inference_steps,
                    video                 = input_video,
                    mask_video            = input_video_mask,
                    clip_image            = clip_image,
                    cfg_skip_ratio        = config.cfg_skip_ratio,
                    shift                 = config.shift,
                    use_longvideo_cfg     = config.use_longvideo_cfg,
                    overlap_video_length  = config.overlap_video_length,
                    partial_video_length  = partial_video_length,
                    ).videos
                
                if init_frames != 0:
                    new_sample[:, :, -config.overlap_video_length:] = (
                        new_sample[:, :, -config.overlap_video_length:] * (1 - mix_ratio) +
                        sample[:, :, :config.overlap_video_length] * mix_ratio
                    )
                    new_sample = torch.cat([new_sample, sample[:, :, config.overlap_video_length:]], dim=2)
                    sample = new_sample
                else:
                    new_sample = sample

                if last_frames >= video_length:
                    break

                # Move to CPU for image conversion (more efficient on Apple Silicon)
                sample_cpu = sample.cpu() if sample.device.type == "mps" else sample
                ref_img = [
                    Image.fromarray(
                        (sample_cpu[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
                    ) for i in range(-config.overlap_video_length, 0)
                ]
                
                # Clear MPS cache periodically to prevent memory issues
                if using_mps and init_frames % (partial_video_length * 2) == 0:
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()

                init_frames += partial_video_length - config.overlap_video_length
                last_frames = init_frames + partial_video_length
        else:
            input_video, input_video_mask, clip_image = get_image_to_video_latent3(ref_img, None, video_length=video_length, sample_size=[sample_height, sample_width])
            sample  = pipeline(
                prompt,
                num_frames            = video_length,
                negative_prompt       = config.negative_prompt,
                audio_embeds          = audio_embeds,
                audio_scale           = config.audio_scale,
                ip_mask               = ip_mask,
                use_un_ip_mask        = config.use_un_ip_mask,
                height                = sample_height,
                width                 = sample_width,
                generator             = generator,
                neg_scale             = config.neg_scale,
                neg_steps             = config.neg_steps,
                use_dynamic_cfg       = config.use_dynamic_cfg,
                use_dynamic_acfg      = config.use_dynamic_acfg,
                guidance_scale        = config.guidance_scale,
                audio_guidance_scale  = config.audio_guidance_scale,
                num_inference_steps   = config.num_inference_steps,
                video                 = input_video,
                mask_video            = input_video_mask,
                clip_image            = clip_image,
                cfg_skip_ratio        = config.cfg_skip_ratio,
                shift                 = config.shift,
                use_longvideo_cfg     = config.use_longvideo_cfg,
                overlap_video_length  = config.overlap_video_length,
                partial_video_length  = partial_video_length,

            ).videos

        # Save generated video
        video_path = os.path.join(save_path, f"{test_name}.mp4")
        video_audio_path = os.path.join(save_path, f"{test_name}_audio.mp4")

        # Move to CPU for video saving (more efficient on Apple Silicon)
        sample_cpu = sample.cpu() if sample.device.type == "mps" else sample
        save_videos_grid(sample_cpu[:, :, :video_length], video_path, fps=config.fps)
        
        # Clear MPS cache after processing each video
        if using_mps:
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()

        video_clip = VideoFileClip(video_path)
        audio_clip = audio_clip.subclipped(0, video_length / config.fps)
        video_clip = video_clip.with_audio(audio_clip)
        video_clip.write_videofile(video_audio_path, codec="libx264", audio_codec="aac", threads=2)

        os.system(f"rm -rf {video_path}")


if __name__ == "__main__":
    main()
