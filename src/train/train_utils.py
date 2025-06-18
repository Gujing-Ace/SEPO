import transformers
import torch
import logging
import os


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """安全保存全参数训练模型，控制精度并避免冗余数据"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 获取模型引用
        model = trainer.model
        
        # 检查模型参数精度
        param_dtype = next(model.parameters()).dtype
        print(f"[INFO] 模型参数当前精度: {param_dtype}")
        
        # 保存前检查是否需要转换精度
        save_dtype = None
        if param_dtype == torch.float16 or param_dtype == torch.bfloat16:
            # 询问用户是否要保持低精度保存
            print("[WARN] 检测到模型使用低精度({param_dtype})训练，"
                  "保存为FP32会导致体积翻倍，将会以低精度保存")
            # 实际应用中可改为参数控制，例如: keep_low_precision=training_args.keep_low_precision
            keep_low_precision = True
            
            if keep_low_precision:
                save_dtype = param_dtype
                print(f"[INFO] 将保持{param_dtype}精度保存以减小体积")
            else:
                print("[INFO] 将保存为FP32精度")
        
        # 保存模型
        if save_dtype:
            # 保存为低精度
            model_to_save = model.half() if save_dtype == torch.float16 else model.bfloat16()
            model_to_save.save_pretrained(
                output_dir,
                safe_serialization=True,
                max_shard_size="10GB"
            )
            # 恢复原模型精度
            del model_to_save
            torch.cuda.empty_cache()
        else:
            # 直接保存（默认FP32）
            model.save_pretrained(
                output_dir,
                safe_serialization=True,
                max_shard_size="10GB"
            )
        
        model.config.save_pretrained(output_dir)
        print(f"[INFO] 模型配置已保存到 {output_dir}")
        
        # 保存训练状态（不含优化器）
        trainer.save_state()
        print(f"[INFO] 训练状态已保存到 {output_dir}")
        
        # 打印最终保存大小
        total_size = sum(os.path.getsize(f"{output_dir}/{f}") for f in os.listdir(output_dir) if os.path.isfile(f"{output_dir}/{f}"))
        print(f"[SUCCESS] 模型保存完成，总大小: {total_size / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"[ERROR] 模型保存失败: {e}")
        raise