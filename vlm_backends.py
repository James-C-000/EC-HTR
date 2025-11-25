#!/usr/bin/env python3
"""
VLM Integration Module for EC historical weather scan processing

This module provides different VLM backend implementations that can be plugged
into the main processor. I've experimented with:
- Ollama (easy but limited configurability)
- vLLM (optimized but high VRAM requirements)
- Transformers (flexible but OOM-prone)
- llama.cpp (buggy with Qwen3-VL as of testing)

This module provides a unified interface for these backends.

Author: James C. Caldwell
"""

import os
import base64
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM backends."""
    model_name: str = "qwen2.5-vl:7b"
    max_tokens: int = 1024
    temperature: float = 0.1  # Low temperature for factual extraction
    timeout_seconds: int = 60
    
    # Image preprocessing
    max_image_size: int = 2048  # Max dimension in pixels
    image_quality: int = 85  # JPEG quality if conversion needed
    
    # Performance tuning
    min_pixels: Optional[int] = None  # For models that support this
    max_pixels: Optional[int] = None


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
    
    @abstractmethod
    def extract_text(self, image_path: str, prompt: str) -> str:
        """
        Extract text from an image using the VLM.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt to send to the VLM
            
        Returns:
            Extracted text from the VLM response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available/configured."""
        pass
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image if needed (resize, convert format).
        
        Returns path to processed image (may be same as input).
        """
        # Default: no preprocessing
        return image_path
    
    def _load_image_base64(self, image_path: str) -> str:
        """Load image and encode as base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


class OllamaBackend(VLMBackend):
    """
    Ollama backend for VLM inference.
    
    Pros: Easy setup, good for local development
    Cons: Limited configurability (no min_pixels/max_pixels)
    
    Usage:
        backend = OllamaBackend(VLMConfig(model_name='qwen2.5-vl:7b'))
        text = backend.extract_text('/path/to/image.png', 'Extract text from this image')
    """
    
    def __init__(self, config: VLMConfig, host: str = "http://localhost:11434"):
        super().__init__(config)
        self.host = host
        self._client = None
    
    def _get_client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")
        return self._client
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            client = self._get_client()
            models = client.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            # Check if our model is available (handle version tags)
            base_model = self.config.model_name.split(':')[0]
            return any(base_model in name for name in model_names)
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    def extract_text(self, image_path: str, prompt: str) -> str:
        """Extract text using Ollama."""
        client = self._get_client()
        
        start_time = time.time()
        
        try:
            response = client.chat(
                model=self.config.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens,
                }
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"Ollama inference took {elapsed:.2f}s")
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            raise


class TransformersBackend(VLMBackend):
    """
    Hugging Face Transformers backend for VLM inference.
    
    Pros: Flexible, good for experimentation
    Cons: Less optimized, OOM-prone on limited VRAM
    
    Usage:
        backend = TransformersBackend(VLMConfig(model_name='Qwen/Qwen2.5-VL-7B-Instruct'))
        text = backend.extract_text('/path/to/image.png', 'Extract text')
    """
    
    # Map friendly names to HuggingFace model IDs
    MODEL_MAP = {
        'qwen2.5-vl:7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'qwen2.5-vl:3b': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'qwen3-vl:8b': 'Qwen/Qwen3-VL-8B-Instruct',
        'florence-2-large': 'microsoft/Florence-2-large',
    }
    
    def __init__(self, config: VLMConfig, device: str = "cuda"):
        super().__init__(config)
        self.device = device
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load model and processor."""
        if self._model is not None:
            return
            
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("transformers package not installed. Run: pip install transformers torch")
        
        # Get HuggingFace model ID
        model_id = self.MODEL_MAP.get(
            self.config.model_name.lower(),
            self.config.model_name  # Assume it's already a HF model ID
        )
        
        logger.info(f"Loading model: {model_id}")
        
        # Load with automatic device mapping for large models
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded on {self.device}")
    
    def is_available(self) -> bool:
        """Check if transformers and torch are available."""
        try:
            import transformers
            import torch
            return torch.cuda.is_available() or self.device == "cpu"
        except ImportError:
            return False
    
    def extract_text(self, image_path: str, prompt: str) -> str:
        """Extract text using Transformers."""
        self._load_model()
        
        from PIL import Image
        import torch
        
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize if too large (to avoid OOM)
        max_dim = max(image.size)
        if max_dim > self.config.max_image_size:
            scale = self.config.max_image_size / max_dim
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)
            logger.debug(f"Resized image from {max_dim} to {max(new_size)}")
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self._model.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )
        
        # Decode response
        response = self._processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        elapsed = time.time() - start_time
        logger.debug(f"Transformers inference took {elapsed:.2f}s")
        
        return response


class VLLMBackend(VLMBackend):
    """
    vLLM backend for high-performance inference.
    
    Pros: Highly optimized, supports batching
    Cons: Requires significant VRAM, complex setup
    
    Usage:
        backend = VLLMBackend(VLMConfig(model_name='Qwen/Qwen2.5-VL-7B-Instruct'))
        text = backend.extract_text('/path/to/image.png', 'Extract text')
    """
    
    def __init__(
        self,
        config: VLMConfig,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        super().__init__(config)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
    
    def _load_model(self):
        """Lazy load vLLM engine."""
        if self._llm is not None:
            return
            
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm package not installed. Run: pip install vllm")
        
        logger.info(f"Loading vLLM model: {self.config.model_name}")
        
        self._llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self._sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
    
    def is_available(self) -> bool:
        """Check if vLLM is available."""
        try:
            import vllm
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def extract_text(self, image_path: str, prompt: str) -> str:
        """Extract text using vLLM."""
        self._load_model()
        
        from PIL import Image
        
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # vLLM multimodal input format
        multimodal_data = {
            "image": image
        }
        
        # Generate
        outputs = self._llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": multimodal_data,
            },
            self._sampling_params,
        )
        
        response = outputs[0].outputs[0].text
        
        elapsed = time.time() - start_time
        logger.debug(f"vLLM inference took {elapsed:.2f}s")
        
        return response


class Florence2Backend(VLMBackend):
    """
    Florence-2 backend for vision-language tasks.
    
    Florence-2 is a specialized vision encoder-decoder that may work better
    for document understanding tasks than general VLMs.
    
    Usage:
        backend = Florence2Backend(VLMConfig(model_name='microsoft/Florence-2-large'))
        text = backend.extract_text('/path/to/image.png', '<OCR>')
    """
    
    # Florence-2 task prompts
    TASK_PROMPTS = {
        'ocr': '<OCR>',
        'ocr_with_region': '<OCR_WITH_REGION>',
        'caption': '<CAPTION>',
        'detailed_caption': '<DETAILED_CAPTION>',
        'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
    }
    
    def __init__(self, config: VLMConfig, task: str = 'ocr'):
        super().__init__(config)
        self.task = task
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load Florence-2 model."""
        if self._model is not None:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("transformers package not installed")
        
        model_id = "microsoft/Florence-2-large"
        
        logger.info(f"Loading Florence-2: {model_id}")
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
    
    def is_available(self) -> bool:
        """Check if Florence-2 dependencies are available."""
        try:
            import transformers
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def extract_text(self, image_path: str, prompt: str) -> str:
        """Extract text using Florence-2."""
        self._load_model()
        
        from PIL import Image
        import torch
        
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Use task prompt or custom prompt
        task_prompt = self.TASK_PROMPTS.get(self.task, prompt)
        
        inputs = self._processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self._model.device, torch.float16)
        
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_tokens,
                num_beams=3,
            )
        
        response = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Parse Florence-2 response format
        if task_prompt in response:
            response = response.replace(task_prompt, '').strip()
        
        elapsed = time.time() - start_time
        logger.debug(f"Florence-2 inference took {elapsed:.2f}s")
        
        return response


def create_vlm_callback(
    backend_type: str = "ollama",
    config: Optional[VLMConfig] = None,
    **kwargs
):
    """
    Factory function to create a VLM callback for the main processor.
    
    Args:
        backend_type: One of "ollama", "transformers", "vllm", "florence2"
        config: VLMConfig instance (uses defaults if None)
        **kwargs: Additional arguments passed to the backend constructor
    
    Returns:
        A callback function that takes (image_path, prompt) and returns extracted text
    """
    if config is None:
        config = VLMConfig()
    
    backends = {
        'ollama': OllamaBackend,
        'transformers': TransformersBackend,
        'vllm': VLLMBackend,
        'florence2': Florence2Backend,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")
    
    backend_class = backends[backend_type]
    backend = backend_class(config, **kwargs)
    
    if not backend.is_available():
        logger.warning(f"Backend {backend_type} may not be fully available. Check dependencies.")
    
    def callback(image_path: str, prompt: str) -> str:
        return backend.extract_text(image_path, prompt)
    
    return callback


# Example integration with main processor
if __name__ == "__main__":
    # Example: Test with Ollama backend
    config = VLMConfig(
        model_name='qwen2.5-vl:7b',
        max_tokens=1024,
        temperature=0.1
    )
    
    # Try to create Ollama backend
    try:
        callback = create_vlm_callback('ollama', config)
        print("Ollama backend created successfully")
        
        # Test availability
        backend = OllamaBackend(config)
        if backend.is_available():
            print("Ollama is available and model is loaded")
        else:
            print("Ollama not available - make sure it's running")
    except ImportError as e:
        print(f"Could not create Ollama backend: {e}")
    
    # Example usage with main processor:
    """
    from ec_vlm_processor import ECProcessor
    
    # Create processor
    processor = ECProcessor(
        inventory_2022_path='EC_Historical_Weather_Station_inventory_2022.csv',
        inventory_2014_path='EC_Historical_Weather_Station_inventory_2014_01.csv',
        checkpoint_dir='./checkpoints'
    )
    
    # Create VLM callback
    vlm_callback = create_vlm_callback('ollama', VLMConfig(model_name='qwen2.5-vl:7b'))
    
    # Process directory
    stats = processor.process_directory(
        root_dir='/path/to/scans',
        vlm_callback=vlm_callback
    )
    """
