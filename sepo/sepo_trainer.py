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
                'lm_head' in name or
                'input_layernorm.weight' in name or
                'post_attention_layernorm.weight' in name) and len(param.shape) == 2:
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
    def __init__(self, *args, topk_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable hidden states output only
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = False  # Disable due to model implementation issues
        
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
        outputs_rejected = self.noise_injector(**model_inputs)

        # 移除噪声注入，获取 chosen 样本的输出
        original_model = self.noise_injector.model
        outputs_chosen = original_model(**model_inputs)
        
        # 调试输出 - 检查模型输出结构
        # print("\n=== Model Output Structure Debug ===")
        # print(f"Output object type: {type(outputs_chosen)}")
        # print(f"Output attributes: {dir(outputs_chosen)}")
        # if hasattr(outputs_chosen, 'last_hidden_state'):
        #     print(f"last_hidden_state shape: {outputs_chosen.last_hidden_state.shape}")
        # if hasattr(outputs_chosen, 'hidden_states'):
        #     print(f"hidden_states length: {len(outputs_chosen.hidden_states) if outputs_chosen.hidden_states else 0}")
        # if hasattr(outputs_chosen, 'attentions'):
        #     print(f"attentions length: {len(outputs_chosen.attentions) if outputs_chosen.attentions else 0}")
        # print("==================================\n")

        # 2. Calculate DPO loss
        logits_chosen = outputs_chosen.logits
        logits_rejected = outputs_rejected.logits
        labels = inputs.get('labels')

        def get_logprobs(logits, labels):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            vocab_size = log_probs.size(-1)
            # 将超出词汇表大小或为负数的索引替换为 IGNORE_INDEX
            shift_labels = torch.where((shift_labels >= vocab_size) | (shift_labels < 0), IGNORE_INDEX, shift_labels)

            # 创建一个掩码，标记非 IGNORE_INDEX 的位置
            valid_mask = shift_labels != IGNORE_INDEX
            valid_log_probs = log_probs[valid_mask]
            valid_shift_labels = shift_labels[valid_mask]

            # 初始化一个全为 0 的结果张量
            result = torch.zeros_like(shift_labels, dtype=log_probs.dtype, device=log_probs.device)

            if valid_shift_labels.numel() > 0:
                gathered = valid_log_probs.gather(-1, valid_shift_labels.unsqueeze(-1)).squeeze(-1)
                result[valid_mask] = gathered

            return result

        log_probs_chosen = get_logprobs(logits_chosen, labels)
        log_probs_rejected = get_logprobs(logits_rejected, labels)

        # ce loss
        logits = outputs_chosen.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX
        )
        # DPO 损失计算的超参数
        beta = 0.1

        # 计算 DPO 损失
        losses = -F.logsigmoid(beta * (log_probs_chosen - log_probs_rejected))
        dpo_loss = losses.mean()

        # 3. Calculate enhanced multimodal alignment reward
        mm_reward = 0
        if "pixel_values" in inputs and hasattr(outputs_chosen, 'hidden_states') and outputs_chosen.hidden_states:
            # print("MMReward from hidden_states Enabled!\n")
            # Get last layer hidden states
            hidden_states = outputs_chosen.hidden_states[-1]  # [batch, seq_len, dim]
            
            # Use first token as image feature, last token as text feature
            image_features = hidden_states[:, 0, :]  # First token
            text_features = hidden_states[:, -1, :]  # Last token
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute alignment score (avoiding cosine similarity)
            # Using negative L2 distance between normalized features
            mm_reward = -torch.norm(image_features - text_features, p=2, dim=-1).mean()

        # 4. Combine losses with configurable weights
        total_loss = dpo_loss * 0.6 - mm_reward * 0.2 + ce_loss * 0.2  

        # 5. Update rank adaptor based on combined performance
        self.rank_adaptor.update_rank((1 - dpo_loss.item() + mm_reward) / 2)

        if return_outputs:
            return (total_loss, outputs_chosen) if total_loss is not None else (None, outputs_chosen)
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
