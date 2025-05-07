import torch
import torch.nn as nn
from transformers import Trainer
import re
from typing import Dict, Union, List, Optional, Any

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
                'v_proj.weight' in name) and len(param.shape) == 2:
                safe_name = re.sub(r'\.', '_', name)
                # For Qwen models, shapes might be [dim_out, dim_in] or [dim_in, dim_out]
                # Assuming param.shape gives (dim_out, dim_in) for linear layers if not transposed
                # Or (dim_in, dim_out) if that's the convention. Let's assume (rows, cols)
                # If U is (dim_in, rank) and V is (rank, dim_out), then U@V is (dim_in, dim_out)
                # If param is (dim_out, dim_in), then noise should be (dim_out, dim_in)
                # So U (dim_out, rank), V (rank, dim_in) -> U@V (dim_out, dim_in)
                dim_rows, dim_cols = param.shape
                U = nn.Parameter(torch.randn(dim_rows, self.rank)) * 0.02
                V = nn.Parameter(torch.randn(self.rank, dim_cols)) * 0.02
                self.noise_params[safe_name + '_U'] = U
                self.noise_params[safe_name + '_V'] = V

    def forward(self, input_ids, **kwargs):
        # Store original parameters that will be modified
        original_params = {}

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if ('q_proj.weight' in name or 
                    'k_proj.weight' in name or 
                    'v_proj.weight' in name) and len(param.shape) == 2:
                    safe_name = re.sub(r'\.', '_', name)
                    U = self.noise_params.get(safe_name + '_U')
                    V = self.noise_params.get(safe_name + '_V')
                    if U is not None and V is not None:
                        noise = self.alpha * (U @ V)
                        original_params[name] = param.clone() # Save before adding noise
                        param.add_(noise)

        outputs = self.model(input_ids=input_ids, **kwargs)

        # Restore parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.copy_(original_params[name]) # Restore from saved copy
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
        
        if len(self.reward_history) == self.window and len(self.reward_history) >= 20: # Ensure enough history for trend
            # Compare last 10 vs previous 10 in the window
            trend = sum(self.reward_history[-10:]) - sum(self.reward_history[-20:-10]) 
            if trend < 0: # If reward is decreasing
                self.current_rank = min(self.current_rank + 2, self.max_rank)
            else: # If reward is increasing or stable
                self.current_rank = max(self.current_rank - 1, 1) # Min rank is 1

class SEPOTrainer(Trainer):
    def __init__(self, *args, topk_ratio: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_injector = LowRankNoiseInjection(self.model)
        self.rank_adaptor = RankAdaptor()
        self.topk_ratio = topk_ratio
        # Ensure model is on the correct device for noise_injector if not handled by Trainer
        self.noise_injector.to(self.model.device)


    def _prepare_model_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare dict of model inputs to be passed to model.forward().
        This is a simplified version, actual Qwen-VL might need more specific keys.
        """
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }
        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs: # Specific to some Qwen-VL versions
             model_inputs["image_grid_thw"] = inputs["image_grid_thw"]
        # Add other potential multimodal inputs if present in your data
        # e.g., pixel_values_videos, video_grid_thw, second_grid_ts from dpo_trainer
        if "pixel_values_videos" in inputs:
            model_inputs["pixel_values_videos"] = inputs["pixel_values_videos"]
        if "video_grid_thw" in inputs:
            model_inputs["video_grid_thw"] = inputs["video_grid_thw"]
        if "second_grid_ts" in inputs:
            model_inputs["second_grid_ts"] = inputs["second_grid_ts"]
        
        # Filter out None values
        return {k: v for k, v in model_inputs.items() if v is not None}

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 动态调整噪声秩
        self.noise_injector.rank = self.rank_adaptor.current_rank
        
        # Prepare inputs for the model
        model_inputs = self._prepare_model_inputs(inputs)

        # 2. 生成正/负样本
        # Ensure noise_injector's parameters are on the same device as the model
        self.noise_injector.to(model_inputs['input_ids'].device)

        outputs_pos = self.noise_injector(**model_inputs)
        
        with torch.no_grad(): # Temporarily change alpha for negative sample
            original_alpha = self.noise_injector.alpha
            self.noise_injector.alpha = -original_alpha
            # Need to re-run noise injection for negative sample
            # The forward pass of noise_injector itself applies and removes noise
            outputs_neg = self.noise_injector(**model_inputs)
            self.noise_injector.alpha = original_alpha # Restore alpha

        # 3. 计算奖励差
        # Assuming logits are of shape (batch_size, seq_len, vocab_size)
        # The original code used logits_pos.logits[:, -1, :], which is the logit for the *last token prediction*
        # This might be a simplification. A more common approach for sequence rewards is to sum log-probs.
        # For now, sticking to the user's original reward formulation.
        logits_pos = outputs_pos.logits
        logits_neg = outputs_neg.logits

        # Ensure logits are on the same device for subtraction
        if logits_pos.device != logits_neg.device:
            logits_neg = logits_neg.to(logits_pos.device)

        # Example: use mean logit value of the last token prediction
        # This is a placeholder; a better reward signal is likely needed.
        # reward_diff = (logits_pos[:, -1, :] - logits_neg[:, -1, :]).mean()
        
        # Let's use the sum of log-probabilities of the chosen sequence as a base,
        # and the difference due to noise as the reward.
        # For simplicity, let's use the mean of all logits in the sequence as a proxy.
        # This is highly speculative and likely needs refinement based on SEPO paper.
        # Using the original simple version for now:
        reward_diff = (logits_pos[:, -1, :] - logits_neg[:, -1, :]).mean()

        # Update rank adaptor based on this reward_diff
        # The rank adaptor expects a single scalar reward.
        self.rank_adaptor.update_rank(reward_diff.item())

        loss = -reward_diff  # 目标：最大化奖励差

        if return_outputs:
            return (loss, outputs_pos) if loss is not None else (None, outputs_pos)
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None):
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        """
        model.train()
        inputs = self._prepare_inputs(inputs) # Moves inputs to model.device

        if getattr(self, "use_cuda_amp", False) or getattr(self, "use_cpu_amp", False):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.accelerator.sync_gradients:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            # Refer to https://github.com/huggingface/transformers/pull/20755
            if getattr(self, "do_grad_scaling", False): # For AMP
                 loss = self.scaler.scale(loss)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
        elif getattr(self, "do_grad_scaling", False): # For AMP
            self.scaler.scale(loss).backward()
        elif self.use_apex: # For apex AMP
            with self.accelerator.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else: # Vanilla backward
            self.accelerator.backward(loss)
        
        # --- 梯度竞争开始 ---
        if self.args.gradient_accumulation_steps == 1 or self.state.global_step % self.args.gradient_accumulation_steps == 0:
            # Only apply gradient competition when an optimizer step is about to happen
            with torch.no_grad():
                sensitivities = {}
                param_groups = [] # Store params to decide threshold later
                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        sensitivities[name] = param.grad.abs().mean().item()
                        param_groups.append(param)
                
                if sensitivities: # if there are any gradients
                    all_sensitivities_values = torch.tensor(list(sensitivities.values()), device=param_groups[0].device if param_groups else 'cpu')
                    if len(all_sensitivities_values) > 0:
                        threshold = torch.quantile(all_sensitivities_values, 1.0 - self.topk_ratio)
                        for name, param in model.named_parameters():
                            if param.grad is not None and param.requires_grad:
                                if sensitivities.get(name, 0) < threshold.item():
                                    param.grad.zero_()
        # --- 梯度竞争结束 ---
        
        return loss.detach()
