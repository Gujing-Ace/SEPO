import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

# Adjust path to import from parent directory's src
# This assumes train_sepo.py is run from within the 'sepo' directory or the project root is in PYTHONPATH
# For robust execution, ensure PYTHONPATH includes the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    AutoProcessor,
    AutoTokenizer, # Qwen processor includes tokenizer
    Qwen2_5_VLForConditionalGeneration, # Specific model class
    HfArgumentParser,
    BitsAndBytesConfig,
    TrainingArguments as HFTrainingArguments # Base HuggingFace TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# SEPO specific imports
from .sepo_trainer import SEPOTrainer
from .sepo_data import make_sepo_data_module

# Attempt to import argument classes from src.train.params
# If these are not found, they might need to be copied or redefined in sepo_params.py
try:
    from src.train.params import ModelArguments as BaseModelArguments, DataArguments
    from src.train.params import TrainingArguments as SrcTrainingArguments # Use this as a base for SEPO specific args
    # Define default tokens if import fails
    DEFAULT_PAD_TOKEN = "<|endoftext|>"
    DEFAULT_EOS_TOKEN = "<|endoftext|>"
    DEFAULT_BOS_TOKEN = "<|endoftext|>"
    DEFAULT_UNK_TOKEN = "<|endoftext|>"
    try:
        from src.train.train_utils import safe_save_model_for_hf_trainer
    except ImportError:
        pass
    # Monkey patching for Qwen models if needed (copied from original train_dpo.py)
    try:
        from src.train.monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl # If liger is used
    except ImportError:
        pass
except ImportError as e:
    print(f"Could not import from src.train: {e}. Ensure PYTHONPATH is set correctly or copy necessary files.")
    # Define minimal fallbacks if imports fail
    @dataclass
    class BaseModelArguments:
        model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-Instruct-3B") # Default to a smaller Qwen VL model

    @dataclass
    class DataArguments:
        data_path: str = field(default=None, metadata={"help": "Path to the training data."})
        image_folder: Optional[str] = field(default=None)
        # Add other fields from src.train.params.DataArguments as needed
        image_min_pixels: Optional[int] = field(default=3136)
        image_max_pixels: Optional[int] = field(default=12845056)
        video_min_pixels: Optional[int] = field(default=100352)
        video_max_pixels: Optional[int] = field(default=602112)
        image_resized_width: int = field(default=None)
        image_resized_height: int = field(default=None)
        video_resized_width: int = field(default=None)
        video_resized_height: int = field(default=None)
        fps: float = 1.0

    @dataclass
    class SrcTrainingArguments(HFTrainingArguments): # Inherit from HF base
        cache_dir: Optional[str] = field(default=None)
        optim: str = field(default="adamw_torch")
        max_seq_length: int = field(default=2048) # Adjust as needed
        lora_enable: bool = field(default=False)
        lora_rank: int = field(default=64)
        lora_alpha: int = field(default=16)
        lora_dropout: float = field(default=0.05)
        lora_bias: str = field(default="none")
        bits: int = field(default=16) # For quantization
        double_quant: bool = field(default=True)
        quant_type: str = field(default="nf4")
        freeze_vision_tower: bool = field(default=False)
        freeze_llm: bool = field(default=False)
        use_liger: bool = field(default=False) # Default to False if liger is not set up
        disable_flash_attn2: bool = field(default=False)


    DEFAULT_PAD_TOKEN = "<|endoftext|>" # Common pad token
    DEFAULT_EOS_TOKEN = "<|endoftext|>"
    DEFAULT_BOS_TOKEN = "<|endoftext|>" # Or specific BOS for model
    DEFAULT_UNK_TOKEN = "<|endoftext|>"
    
    def safe_save_model_for_hf_trainer(trainer, output_dir: str):
        """HF save model utility"""
        trainer.save_model(output_dir)

    # Dummy monkey patching functions if import fails
    def replace_qwen2_5_with_mixed_modality_forward(use_liger=False): pass
    def replace_qwen_2_with_mixed_modality_forward(use_liger=False): pass
    def apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False): pass
    def apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False): pass


@dataclass
class SEPOTrainingArguments(SrcTrainingArguments): # Inherit from the one in src.train.params
    topk_ratio: float = field(default=0.2, metadata={"help": "Top-k ratio for gradient competition in SEPO."})
    # Add any other SEPO specific training arguments here

def train():
    parser = HfArgumentParser((BaseModelArguments, DataArguments, SEPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Liger kernel and monkey patching (optional, based on original script)
    if training_args.use_liger:
        if "Qwen2.5" in model_args.model_id:
            replace_qwen2_5_with_mixed_modality_forward(use_liger=True)
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
        else: # Assuming Qwen2-VL
            replace_qwen_2_with_mixed_modality_forward(use_liger=True)
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
    
    # Quantization config (for 8GB VRAM, 4-bit quantization is recommended)
    if training_args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
        )
    elif training_args.bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else: # 16-bit or 32-bit
        quantization_config = None

    # Load model
    # For 8GB VRAM, consider a smaller model version if Qwen2.5-VL-3B is too large even with 4-bit
    # Example: model_args.model_id = "Qwen/Qwen-VL-Chat" or a smaller variant if available
    # User specified Qwen2.5-VL-3B-Instruct in sepo/models, so we use that.
    # The path should be relative to project root or absolute.
    # Assuming model_args.model_id is "sepo/models/Qwen2.5-VL-3B-Instruct" or similar
    # If it's a HuggingFace ID, it will download. If local path, it will load from there.
    
    # Correct model path based on user's file structure
    # The model is at 'sepo/models/Qwen2.5-VL-3B-Instruct' relative to project root
    # If model_args.model_id is not an absolute path, prepend with project root.
    # However, HuggingFace from_pretrained usually handles this if model_id is a valid path.
    # For safety, let's ensure the path is what's intended if it's a local model.
    # The user has it under sepo/models/
    # If model_args.model_id is just "Qwen2.5-VL-3B-Instruct", it might try to download.
    # We should use the local path.
    
    # Default model_id if not provided or to ensure local path:
    # This should be set via command line: --model_id sepo/models/Qwen2.5-VL-3B-Instruct
    # For testing, we can hardcode it if needed, but CLI arg is better.
    # model_load_path = model_args.model_id
    # if not os.path.isdir(model_load_path): # If not a full path, assume it's relative to project root
    #     model_load_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_args.model_id)


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available for 8GB
        quantization_config=quantization_config,
        device_map={"":torch.cuda.current_device()} if quantization_config else "auto", # device_map for quantization
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else None
    )

    # Load processor (includes tokenizer for Qwen-VL)
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    # Qwen VL uses <|endoftext|> as pad token by default.
    # Ensure pad token is set if model doesn't have one or if it's different.
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token # Common practice
    
    # LoRA configuration (recommended for 8GB VRAM)
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common modules for LLMs
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            # modules_to_save = ["wte", "lm_head"] # if you want to train the embeddings and output layer
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Freeze parameters if needed (after LoRA application)
    if training_args.freeze_llm and not training_args.lora_enable: # Full finetuning with llm frozen doesn't make sense
        for name, param in model.named_parameters():
            if "visual" not in name and "vision_tower" not in name: # crude way to identify LLM parts
                param.requires_grad = False
    if training_args.freeze_vision_tower:
        for name, param in model.named_parameters():
            if "visual" in name or "vision_tower" in name:
                param.requires_grad = False
    
    # Data module
    # User data path: /mnt/e/Projects/Qwen2-VL-Finetune/sepo/data/RLAIF_Sample/DPO_format_for_Qwen/dpo.json
    # This should be passed via --data_path
    # Image folder: /mnt/e/Projects/Qwen2-VL-Finetune/sepo/data/RLAIF_Sample/DPO_format_for_Qwen/images
    # This should be passed via --image_folder
    data_module = make_sepo_data_module(
        model_id=model_args.model_id, 
        processor=processor, 
        data_args=data_args
    )

    # Initialize SEPOTrainer
    # For 8GB VRAM, set per_device_train_batch_size=1, gradient_accumulation_steps as high as possible (e.g., 8, 16, 32)
    # training_args should be updated with these defaults if not provided by user
    # Example:
    # training_args.per_device_train_batch_size = 1
    # training_args.gradient_accumulation_steps = 8 # or 16
    # training_args.fp16 = True # or bf16 if supported
    # training_args.tf32 = True # If using Ampere or newer
    # training_args.optim = "paged_adamw_8bit" # If using bitsandbytes and LoRA for memory saving

    trainer = SEPOTrainer(
        model=model,
        args=training_args,
        topk_ratio=training_args.topk_ratio, # Pass topk_ratio
        **data_module
    )

    # Start training
    trainer.train()

    # Save model
    # trainer.save_state() # Saves trainer state
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
