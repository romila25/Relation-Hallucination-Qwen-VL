# reefknot_qwen.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
class ReefknotQwen(nn.Module):
    """
    Reefknot-compatible wrapper for Qwen LLM.
    Supports DTC patching and multi-modal integration.
    Loads model/tokenizer only once (singleton pattern).
    """
    _global_llm_instance = None
    _global_tokenizer = None
    _global_processor = None

    def __init__(self, model_path, dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Tokenizer caching
        if ReefknotQwen._global_tokenizer is None:
            print(f"[ReefknotQwen] Loading tokenizer from {model_path}...")
            ReefknotQwen._global_tokenizer = AutoTokenizer.from_pretrained(model_path, **model_kwargs)
        
        self.tokenizer = ReefknotQwen._global_tokenizer

        # Model caching
        if ReefknotQwen._global_llm_instance is None:
            print(f"[ReefknotQwen] Loading model from {model_path} (dtype={dtype})...")

            ReefknotQwen._global_llm_instance = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
            
        if ReefknotQwen._global_processor is None:
            print(f"[ReefknotQwen] Loading processor from {model_path} (dtype={dtype})...")
            ReefknotQwen._global_processor = AutoProcessor.from_pretrained(model_path, **model_kwargs)
            
        else:
            print("[ReefknotQwen] Reusing existing model instance")

        self.llm = ReefknotQwen._global_llm_instance
        self.processor = ReefknotQwen._global_processor
        
        # Patch LM head if needed (for DTC)
        if not hasattr(self, "lm_head"):
            self.lm_head = self.llm.get_output_embeddings()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(self, **kwargs):
        return self.llm(**kwargs)

    def generate(self, **kwargs):
        # This will be patched by DTC_function
        return self.llm.generate(**kwargs)

    @property
    def config(self):
        return self.llm.config

    @property
    def device(self):
        return next(self.llm.parameters()).device
    # Optional: convenience for tokenizer encoding
    def encode(self, text, return_tensors="pt"):
        return self.tokenizer(text, return_tensors=return_tensors)

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
