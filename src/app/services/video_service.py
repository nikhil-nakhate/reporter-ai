"""
Service for generating videos from images and audio using echomimic_v3.
"""
import logging
import os
import sys
import math
import gc
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from moviepy import VideoFileClip, AudioFileClip
import librosa

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import echomimic_v3 modules
echomimic_path = Path(__file__).parent.parent.parent / "third_party" / "echomimic_v3"
sys.path.insert(0, str(echomimic_path))

logger = logging.getLogger(__name__)

# Try to import echomimic_v3 modules
ECHOMIMIC_AVAILABLE = False
try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from src.dist import set_multi_gpus_devices
    from src.wan_vae import AutoencoderKLWan
    from src.wan_image_encoder import CLIPModel
    from src.wan_text_encoder import WanT5EncoderModel
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
    ECHOMIMIC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"echomimic_v3 modules not available: {e}")
    ECHOMIMIC_AVAILABLE = False


class VideoService:
    """Service for generating videos from images and audio using echomimic_v3."""
    
    def __init__(self):
        """Initialize the video service and load models at startup for cleaner memory."""
        if not ECHOMIMIC_AVAILABLE:
            logger.error("echomimic_v3 modules not available. Video generation will be disabled.")
            self.pipeline = None
            self.wav2vec_processor = None
            self.wav2vec_model = None
            self.device = None
            self._initialized = False
            return
        
        self.echomimic_path = echomimic_path
        self.config_path = self.echomimic_path / "config" / "config.yaml"
        self.model_name = self.echomimic_path / "models" / "Wan2.1-Fun-V1.1-1.3B-InP"
        self.transformer_path = self.echomimic_path / "models" / "transformer" / "diffusion_pytorch_model.safetensors"
        self.wav2vec_model_dir = self.echomimic_path / "models" / "wav2vec2-base-960h"
        
        # Default configuration (reduced for memory efficiency)
        self.weight_dtype = torch.bfloat16
        self.sample_size = [256, 256]  # Reduced from 512 to save memory
        self.fps = 25
        self.guidance_scale = 4.0
        self.audio_guidance_scale = 2.9
        self.num_inference_steps = 25
        self.negative_prompt = "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands."
        self.seed = 43
        self.partial_video_length = 65  # Reduced from 75 to save memory
        self.overlap_video_length = 8
        
        self.pipeline = None
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self.device = None
        self._initialized = False
        # Default: try to keep the whole pipeline on GPU for speed unless explicitly disabled
        self.use_cpu_offload = os.getenv("VIDEO_SERVICE_CPU_OFFLOAD", "0").lower() in ("1", "true", "yes")
        
        # Initialize models at startup for cleaner memory state
        try:
            logger.info("Initializing video service models at startup...")
            self._initialize()
        except Exception as e:
            logger.error(f"Failed to initialize video service at startup: {e}")
            logger.warning("Video generation will be disabled. App will continue to work without video previews.")
            self.pipeline = None
            self.wav2vec_processor = None
            self.wav2vec_model = None
            self.device = None
            self._initialized = False
    
    def _initialize(self):
        """Initialize the video generation pipeline."""
        if self._initialized:
            return
        
        if not ECHOMIMIC_AVAILABLE:
            raise RuntimeError("echomimic_v3 modules not available")
        
        logger.info("Initializing video generation pipeline...")
        
        # Set up device
        self.device = set_multi_gpus_devices(ulysses_degree=1, ring_degree=1)
        
        # Load configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        cfg = OmegaConf.load(self.config_path)
        
        # Clear CUDA cache before loading models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load models (keep on CPU initially to save memory, following infer.py pattern)
        logger.info("Loading transformer...")
        transformer = WanTransformerAudioMask3DModel.from_pretrained(
            str(self.model_name / cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=False,  # Disabled - not compatible with current diffusers version
        )
        # Explicitly move to CPU to prevent GPU allocation during loading
        transformer = transformer.cpu()
        if self.transformer_path.exists():
            if str(self.transformer_path).endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(str(self.transformer_path))
            else:
                state_dict = torch.load(str(self.transformer_path), map_location='cpu')
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded transformer weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Clear memory after transformer load
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            str(self.model_name / cfg['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
        )
        # Convert to dtype but keep on CPU
        vae = vae.to(dtype=self.weight_dtype).cpu()
        
        # Clear memory after VAE load
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_name / cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        
        # Clear memory after tokenizer load
        gc.collect()
        
        # Aggressive memory clearing before loading large text encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Loading text encoder (this is a large model, may take a moment)...")
        text_encoder = self._load_text_encoder_safe(cfg)
        
        # Clear memory after text encoder load (this is a large model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Loading CLIP image encoder...")
        clip_image_encoder = CLIPModel.from_pretrained(
            str(self.model_name / cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
        )
        # Convert to dtype but keep on CPU
        clip_image_encoder = clip_image_encoder.to(dtype=self.weight_dtype).cpu().eval()
        
        # Clear memory after CLIP load
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load scheduler
        scheduler_cls = FlowMatchEulerDiscreteScheduler
        scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(cfg['scheduler_kwargs'])))
        
        # Create pipeline
        logger.info("Creating pipeline...")
        self.pipeline = WanFunInpaintAudioPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        
        # Prefer full GPU placement for speed; fall back to offload if OOM or explicitly requested
        self._place_pipeline_on_device()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable TeaCache if available
        try:
            coefficients = get_teacache_coefficients(str(self.model_name))
            self.pipeline.transformer.enable_teacache(
                coefficients, self.num_inference_steps, threshold=0.1,
                num_skip_start_steps=5, offload=False
            )
            logger.info("TeaCache enabled.")
        except Exception as e:
            logger.warning(f"TeaCache not available: {e}")
        
        # Load Wav2Vec models
        logger.info("Loading Wav2Vec models...")
        self._load_wav2vec_models()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = True
        logger.info("Video generation pipeline initialized successfully.")
    
    def _load_text_encoder_safe(self, cfg):
        """Load the Wan text encoder with reduced peak CPU memory usage."""
        text_encoder_kwargs = OmegaConf.to_container(cfg["text_encoder_kwargs"])
        text_encoder_path = self.model_name / text_encoder_kwargs.get("text_encoder_subpath", "text_encoder")

        load_kwargs = {"map_location": "cpu"}
        try:
            # Prefer mmap/weights_only when available (PyTorch >= 2.1) to avoid duplicate buffers.
            load_kwargs.update({"weights_only": True, "mmap": True})
            state_dict = torch.load(text_encoder_path, **load_kwargs)
        except TypeError:
            # Older torch without mmap/weights_only support.
            load_kwargs.pop("weights_only", None)
            load_kwargs.pop("mmap", None)
            state_dict = torch.load(text_encoder_path, **load_kwargs)
        except Exception as e:
            logger.warning(f"Memory-friendly torch.load failed ({e}); falling back to standard load...")
            return WanT5EncoderModel.from_pretrained(
                str(text_encoder_path),
                additional_kwargs=text_encoder_kwargs,
                torch_dtype=self.weight_dtype,
                low_cpu_mem_usage=False,
            ).cpu().eval()

        model = WanT5EncoderModel(**filter_kwargs(WanT5EncoderModel, text_encoder_kwargs))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Text encoder loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        # Free state_dict promptly to keep peak RSS lower.
        del state_dict
        gc.collect()

        return model.to(dtype=self.weight_dtype).cpu().eval()
    
    def _enable_cpu_offload(self):
        """Enable CPU offload with sequential preference."""
        logger.info("Enabling CPU offload for the pipeline...")
        try:
            self.pipeline.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offloading enabled.")
        except Exception as e:
            logger.warning(f"Sequential offload failed ({e}), falling back to model CPU offload... ({e})")
            try:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Standard CPU offloading enabled.")
            except Exception as e2:
                logger.warning(f"CPU offload failed ({e2}); keeping current placement.")
    
    def _place_pipeline_on_device(self):
        """Try to keep pipeline on GPU for higher throughput; fallback to CPU offload if needed."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available; staying on CPU with offload.")
            self._enable_cpu_offload()
            return

        if self.use_cpu_offload:
            logger.info("CPU offload explicitly enabled via VIDEO_SERVICE_CPU_OFFLOAD; skipping GPU placement.")
            self._enable_cpu_offload()
            return

        logger.info("Attempting to place pipeline on GPU for better throughput...")
        try:
            self.pipeline.to(device=self.device)
            logger.info("Pipeline placed on GPU; CPU offloading disabled.")
        except RuntimeError as e:
            logger.warning(f"Pipeline GPU placement failed (likely OOM): {e}. Falling back to CPU offload.")
            self._enable_cpu_offload()
    
    def _load_wav2vec_models(self):
        """Load Wav2Vec models for audio feature extraction."""
        if not self.wav2vec_model_dir.exists():
            raise FileNotFoundError(f"Wav2Vec model directory not found: {self.wav2vec_model_dir}")
        
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(str(self.wav2vec_model_dir))
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(str(self.wav2vec_model_dir)).eval()
        self.wav2vec_model.requires_grad_(False)
        # Keep on CPU to save GPU memory
        self.wav2vec_model = self.wav2vec_model.to("cpu")
    
    def _extract_audio_features(self, audio_path: str):
        """Extract audio features using Wav2Vec."""
        sr = 16000
        audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
        input_values = self.wav2vec_processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(next(self.wav2vec_model.parameters()).device)
        features = self.wav2vec_model(input_values).last_hidden_state
        return features.squeeze(0)
    
    def _get_sample_size(self, image: Image.Image, sample_size: Optional[list] = None):
        """Calculate the sample size based on the input image dimensions."""
        if sample_size is None:
            sample_size = self.sample_size
        
        width, height = image.size
        original_area = width * height
        default_area = sample_size[0] * sample_size[1]
        
        if default_area < original_area:
            ratio = math.sqrt(original_area / default_area)
            width = width / ratio // 16 * 16
            height = height / ratio // 16 * 16
        else:
            width = width // 16 * 16
            height = height // 16 * 16
        
        return int(height), int(width)
    
    def _get_ip_mask(self, coords):
        """Create IP mask from coordinates."""
        y1, y2, x1, x2, h, w = coords
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
        mask = mask.reshape(-1)
        return mask.float()
    
    def generate_video(
        self,
        image_path: str,
        audio_path: str,
        prompt: str = "A person speaking",
        output_path: Optional[str] = None,
        sample_size: Optional[list] = [256, 256],
    ) -> str:
        """
        Generate a video from an image and audio file.
        
        Args:
            image_path: Path to the reference image
            audio_path: Path to the audio file
            prompt: Text prompt describing the video
            output_path: Optional output path for the video file
            sample_size: Optional [height, width] for video resolution. Defaults to self.sample_size.
                        Must be multiples of 16. Example: [512, 512], [768, 512], [1024, 768]
            
        Returns:
            Path to the generated video file
        """
        if not ECHOMIMIC_AVAILABLE:
            raise RuntimeError("echomimic_v3 modules not available")
        
        if not self._initialized:
            raise RuntimeError("Video service not initialized. Models should be loaded at startup.")
        
        # Use provided sample_size or default
        current_sample_size = sample_size if sample_size is not None else self.sample_size
        
        # Create output path if not provided
        if output_path is None:
            output_dir = tempfile.gettempdir()
            import uuid
            output_path = os.path.join(output_dir, f"video_{uuid.uuid4().hex}.mp4")
        
        logger.info(f"Generating video from image: {image_path}, audio: {audio_path}")
        logger.info(f"Using sample size: {current_sample_size}")
        
        # Load reference image
        ref_img = Image.open(image_path).convert("RGB")
        
        # Get face mask coordinates
        try:
            y1, y2, x1, x2, h_, w_ = get_mask_coord(image_path)
        except Exception as e:
            logger.warning(f"Could not detect face mask, using default: {e}")
            # Default to center of image
            h_, w_ = ref_img.size[1], ref_img.size[0]
            y1, y2 = h_ // 4, 3 * h_ // 4
            x1, x2 = w_ // 4, 3 * w_ // 4
        
        # Extract audio features
        audio_clip = AudioFileClip(audio_path)
        audio_features = self._extract_audio_features(audio_path)
        audio_embeds = audio_features.unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        
        # Calculate video length
        video_length = int(audio_clip.duration * self.fps)
        video_length = (
            int((video_length - 1) // self.pipeline.vae.config.temporal_compression_ratio * self.pipeline.vae.config.temporal_compression_ratio) + 1
            if video_length != 1 else 1
        )
        
        # Adjust sample size and create IP mask
        sample_height, sample_width = self._get_sample_size(ref_img, current_sample_size)
        downratio = math.sqrt(sample_height * sample_width / h_ / w_)
        coords = (
            int(y1 * downratio // 16), int(y2 * downratio // 16),
            int(x1 * downratio // 16), int(x2 * downratio // 16),
            sample_height // 16, sample_width // 16,
        )
        ip_mask = self._get_ip_mask(coords).unsqueeze(0)
        ip_mask = torch.cat([ip_mask]*3).to(device=self.device, dtype=self.weight_dtype)
        
        # Calculate partial video length (chunking for memory efficiency)
        base_partial_video_length = int((self.partial_video_length - 1) // self.pipeline.vae.config.temporal_compression_ratio * self.pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        
        # Generate video
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        logger.info(f"Generating video with {video_length} frames using chunking (chunk size: {base_partial_video_length}, overlap: {self.overlap_video_length})...")
        
        # Use chunking logic from infer.py to avoid OOM
        # Get clip image for first chunk
        _, _, clip_image = get_image_to_video_latent3(
            ref_img, None, video_length=base_partial_video_length, sample_size=[sample_height, sample_width]
        )
        
        # Generate video in chunks
        init_frames = 0
        partial_video_length = base_partial_video_length
        last_frames = init_frames + partial_video_length
        new_sample = None
        
        # Precompute mix_ratio outside the loop (will be moved to device when needed)
        mix_ratio = torch.linspace(0, 1, steps=self.overlap_video_length).view(1, 1, -1, 1, 1)
        
        with torch.no_grad():
            while init_frames < video_length:
                if last_frames >= video_length:
                    partial_video_length = video_length - init_frames
                    partial_video_length = (
                        int((partial_video_length - 1) // self.pipeline.vae.config.temporal_compression_ratio * self.pipeline.vae.config.temporal_compression_ratio) + 1
                        if video_length != 1 else 1
                    )
                    
                    if partial_video_length <= 0:
                        break
                
                input_video, input_video_mask, _ = get_image_to_video_latent3(
                    ref_img, None, video_length=partial_video_length, sample_size=[sample_height, sample_width]
                )
        
                partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + partial_video_length) * 2]
                
                logger.info(f"Processing chunk: frames {init_frames} to {init_frames + partial_video_length} (chunk length: {partial_video_length})")
                sample = self.pipeline(
                    prompt,
                    num_frames=partial_video_length,
                    negative_prompt=self.negative_prompt,
                    audio_embeds=partial_audio_embeds,
                    audio_scale=1.0,
                    ip_mask=ip_mask,
                    use_un_ip_mask=False,
                    height=sample_height,
                    width=sample_width,
                    generator=generator,
                    neg_scale=1.5,
                    neg_steps=2,
                    use_dynamic_cfg=True,
                    use_dynamic_acfg=True,
                    guidance_scale=self.guidance_scale,
                    audio_guidance_scale=self.audio_guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    cfg_skip_ratio=0,
                    shift=5.0,
                    use_longvideo_cfg=False,
                    overlap_video_length=self.overlap_video_length,
                    partial_video_length=partial_video_length,
                ).videos
                
                if init_frames != 0:
                    # Move mix_ratio to same device as samples
                    mix_ratio_device = mix_ratio.to(device=sample.device, dtype=sample.dtype)
                    new_sample[:, :, -self.overlap_video_length:] = (
                        new_sample[:, :, -self.overlap_video_length:] * (1 - mix_ratio_device) +
                        sample[:, :, :self.overlap_video_length] * mix_ratio_device
                    )
                    new_sample = torch.cat([new_sample, sample[:, :, self.overlap_video_length:]], dim=2)
                    sample = new_sample
                else:
                    new_sample = sample
                
                if last_frames >= video_length:
                    break
                
                # Update reference image from last frames for next chunk
                ref_img = [
                    Image.fromarray(
                        (sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).cpu().numpy().astype(np.uint8)
                    ) for i in range(-self.overlap_video_length, 0)
                ]
                
                init_frames += partial_video_length - self.overlap_video_length
                # Reset partial_video_length to base value for next iteration
                partial_video_length = base_partial_video_length
                last_frames = init_frames + partial_video_length
                
                # Clear CUDA cache between chunks to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save video
        logger.info(f"Saving video to {output_path}")
        save_videos_grid(sample[:, :, :video_length], output_path, fps=self.fps)
        
        # Combine with audio
        video_clip = VideoFileClip(output_path)
        audio_clip = audio_clip.subclipped(0, video_length / self.fps)
        video_clip = video_clip.with_audio(audio_clip)
        
        # Save final video with audio
        final_output_path = output_path.replace(".mp4", "_audio.mp4")
        video_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", threads=2, logger=None)
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Video generated successfully: {final_output_path}")
        return final_output_path

