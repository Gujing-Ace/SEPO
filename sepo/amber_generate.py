import os
import json
import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

def load_model_and_processor(model_path):
    """加载模型和处理器"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    return model, processor

def process_images(model, processor, image_folder):
    """处理图片并生成描述"""
    results = []
    image_files = sorted(glob.glob(os.path.join(image_folder, "AMBER_*.jpg")))
    
    for img_path in tqdm(image_files, desc="Processing images"):
        img_id = int(os.path.basename(img_path).split('_')[1].split('.')[0])
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            description = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            print(description, '\n')
            
            results.append({
                "id": img_id,
                "response": f"The description of {os.path.basename(img_path)} from MLLM: {description}"
            })
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            results.append({
                "id": img_id,
                "response": f"Error processing image: {str(e)}"
            })
    
    return results

if __name__ == "__main__":
    # 配置路径
    model_path = "/root/autodl-tmp/SEPO/sepo/models/Qwen2.5-VL-3B-Instruct"
    image_folder = "/root/autodl-tmp/SEPO/sepo/eval_data/image"
    output_file = "image_descriptions.json"
    
    print("Loading model and processor...")
    model, processor = load_model_and_processor(model_path)
    
    print("Processing images...")
    descriptions = process_images(model, processor, image_folder)
    
    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(descriptions, f, indent=4)
    
    print("Done!")
