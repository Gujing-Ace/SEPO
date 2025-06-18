import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import re
from typing import Dict, Union, List, Optional, Any
from src.train.constants import (
        IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_MESSAGE,
        VISION_START_TOKEN, VISION_END_TOKEN,
        LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN
    )

class LowRankNoiseInjection(nn.Module):
    def __init__(self, model, rank=4, alpha=0.1):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.noise_params = nn.ParameterDict()
        
        for name, param in self.model.named_parameters():
            if ('q_proj.weight' in name or 
                'k_proj.weight' in name or 
                'v_proj.weight' in name or
                'o_proj.weight' in name or
                'gate_proj' in name or
                'up_proj' in name or
                'down_proj' in name or
                'lm_head' in name) and len(param.shape) == 2:
                safe_name = re.sub(r'\.', '_', name)
                dim_rows, dim_cols = param.shape
                U = nn.Parameter(torch.randn(dim_rows, self.rank)) * 0.02
                V = nn.Parameter(torch.randn(self.rank, dim_cols)) * 0.02
                self.noise_params[safe_name + '_U'] = U
                self.noise_params[safe_name + '_V'] = V

    def forward(self, input_ids, **kwargs):
        original_params = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if ('q_proj.weight' in name or 
                    'k_proj.weight' in name or 
                    'v_proj.weight' in name or
                    'o_proj.weight' in name or
                    'gate_proj' in name or
                    'up_proj' in name or
                    'down_proj' in name or
                    'lm_head' in name) and len(param.shape) == 2:
                    safe_name = re.sub(r'\.', '_', name)
                    U = self.noise_params.get(safe_name + '_U')
                    V = self.noise_params.get(safe_name + '_V')
                    if U is not None and V is not None:
                        noise = self.alpha * (U @ V)
                        original_params[name] = param.data.clone()  # Use .data.clone()
                        param.data = (param + noise).data  # Non-inplace operation

        outputs = self.model(input_ids=input_ids, **kwargs)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])  # Use .data.copy_()
        return outputs

class RankAdaptor:
    def __init__(self, initial_rank=4, max_rank=16, reward_window=100):
        self.current_rank = initial_rank
        self.max_rank = max_rank
        self.reward_history = []
        self.window = reward_window

    def update_rank(self, recent_reward):
        self.reward_history.append(recent_reward)
        if len(self.reward_history) > self.window:
            self.reward_history.pop(0)
        
        if len(self.reward_history) == self.window and len(self.reward_history) >= 20:
            trend = sum(self.reward_history[-10:]) - sum(self.reward_history[-20:-10]) 
            if trend < 0:
                self.current_rank = min(self.current_rank + 2, self.max_rank)
            else:
                self.current_rank = max(self.current_rank - 1, 1)

class SEPOTrainer(Trainer):
    def __init__(self, *args, topk_ratio: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_injector = LowRankNoiseInjection(self.model)
        self.rank_adaptor = RankAdaptor()
        self.topk_ratio = topk_ratio
        self.noise_injector.to(self.model.device)

    def _prepare_model_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }
        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
             model_inputs["image_grid_thw"] = inputs["image_grid_thw"]
        if "pixel_values_videos" in inputs:
            model_inputs["pixel_values_videos"] = inputs["pixel_values_videos"]
        if "video_grid_thw" in inputs:
            model_inputs["video_grid_thw"] = inputs["video_grid_thw"]
        if "second_grid_ts" in inputs:
            model_inputs["second_grid_ts"] = inputs["second_grid_ts"]
        return {k: v for k, v in model_inputs.items() if v is not None}

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. Prepare inputs with noise injection
        self.noise_injector.rank = self.rank_adaptor.current_rank
        model_inputs = self._prepare_model_inputs(inputs)
        self.noise_injector.to(model_inputs['input_ids'].device)
        outputs = self.noise_injector(**model_inputs)

        # 2. Calculate base supervised loss (cross entropy with labels)
        logits = outputs.logits
        labels = inputs.get('labels')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX
        )

        # 3. Calculate enhanced multimodal alignment reward
        mm_reward = 0
        if "pixel_values" in inputs and hasattr(outputs, 'cross_attentions'):
            # Use cross attention weights for alignment
            cross_attn = outputs.cross_attentions[-1]  # Last layer attention
            # Average attention across heads and layers
            attn_weights = torch.mean(cross_attn, dim=1)  # [batch, seq_len, num_patches]
            
            # Get most attended regions (top-k attention)
            top_k = min(5, attn_weights.size(-1))  # Use top 5 attended regions
            top_attn = torch.topk(attn_weights, k=top_k, dim=-1)
            
            # Calculate alignment score as mean of top attended regions
            mm_reward = torch.mean(top_attn.values)

        # 4. Combine losses with configurable weights
        total_loss = ce_loss * 0.7 - mm_reward * 0.3  # 70% CE loss, 30% alignment reward

        # 5. Update rank adaptor based on combined performance
        self.rank_adaptor.update_rank((1 - ce_loss.item() + mm_reward) / 2)

        if return_outputs:
            return (total_loss, outputs) if total_loss is not None else (None, outputs)
        return total_loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if getattr(self, "use_cuda_amp", False) or getattr(self, "use_cpu_amp", False):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.accelerator.sync_gradients:
            if getattr(self, "do_grad_scaling", False):
                 loss = self.scaler.scale(loss)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
        elif getattr(self, "do_grad_scaling", False):
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with self.accelerator.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        # 梯度竞争
        if self.args.gradient_accumulation_steps == 1 or self.state.global_step % self.args.gradient_accumulation_steps == 0:
            with torch.no_grad():
                sensitivities = {}
                param_groups = []
                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        sensitivities[name] = param.grad.abs().mean().item()
                        param_groups.append(param)
                
                if sensitivities:
                    all_sensitivities_values = torch.tensor(list(sensitivities.values()), device=param_groups[0].device if param_groups else 'cpu')
                    if len(all_sensitivities_values) > 0:
                        threshold = torch.quantile(all_sensitivities_values, 1.0 - self.topk_ratio)
                        for name, param in model.named_parameters():
                            if param.grad is not None and param.requires_grad:
                                if sensitivities.get(name, 0) < threshold.item():
                                    param.grad.zero_()
        
        return loss.detach()
