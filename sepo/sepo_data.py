import copy
import os
from typing import Dict, Union, Any, List
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image
import re

# Assuming src.train.params and src.train.constants are accessible
# If not, these might need to be copied or adjusted.
# For now, let's assume they can be imported or relevant parts are redefined.
try:
    from src.train.params import DataArguments
    from src.train.constants import (
        IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_MESSAGE,
        VISION_START_TOKEN, VISION_END_TOKEN,
        LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN
    )
    from src.train.data import get_image_info, get_video_info, pad_sequence, replace_image_tokens, llava_to_openai
except ImportError:
    # Fallback if direct import fails, define critical constants here or adjust path
    # This is a common issue when moving files to different directory structures.
    # For a robust solution, consider making 'src' a package or adjusting PYTHONPATH.
    # For now, we'll proceed assuming the try block works. If not, this will need fixing.
    print("Warning: Could not import from src.train.params or src.train.constants. Paths might need adjustment.")
    # Define minimal necessary constants if import fails and they are used directly
    IGNORE_INDEX = -100
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_VIDEO_TOKEN = "<video>"
    DEFAULT_IM_START_TOKEN = "<|im_start|>"
    DEFAULT_IM_END_TOKEN = "<|im_end|>"
    SYSTEM_MESSAGE = "" # Default, can be overridden
    VISION_START_TOKEN = "<|vision_start|>"
    VISION_END_TOKEN = "<|vision_end|>"
    LLAVA_IMAGE_TOKEN = "<image>" # Placeholder, might be different in actual LLaVA
    LLAVA_VIDEO_TOKEN = "<video>" # Placeholder

    # Dummy DataArguments if import fails, real one should be used
    @dataclass
    class DataArguments:
        data_path: str = field(default=None)
        image_folder: Optional[str] = field(default=None)
        image_min_pixels: Optional[int] = field(default=3136)
        image_max_pixels: Optional[int] = field(default=12845056)
        video_min_pixels: Optional[int] = field(default=100352)
        video_max_pixels: Optional[int] = field(default=602112)
        image_resized_width: int = field(default=None)
        image_resized_height: int = field(default=None)
        video_resized_width: int = field(default=None)
        video_resized_height: int = field(default=None)
        fps: float = 1.0
    
    # Dummy functions if import fails
    def get_image_info(image_path, min_pixel, max_pixel, width, height): return torch.randn(3, 224, 224) # Placeholder
    def get_video_info(video_path, min_pixels, max_pixels, width, height, fps): return torch.randn(3, 16, 224, 224), {} # Placeholder
    def pad_sequence(sequences, padding_side='right', padding_value=0):
        max_len = max(s.size(0) for s in sequences)
        padded_sequences = torch.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype, device=sequences[0].device)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :seq.size(0)] = seq
        return padded_sequences
    def replace_image_tokens(input_string, is_video=False): return input_string # Placeholder
    def llava_to_openai(conversations, is_video=False): return conversations # Placeholder


class SEPODataset(Dataset):
    """Dataset for SEPO training. Uses 'chosen' path from DPO-formatted data."""

    def __init__(
        self,
        data_path: Union[str, list],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
    ):
        super(SEPODataset, self).__init__()
        if isinstance(data_path, str):
            try:
                list_data_dict = json.load(open(data_path, "r"))
            except FileNotFoundError:
                raise FileNotFoundError(f"Data file not found: {data_path}")
        else:
            list_data_dict = data_path
        
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        is_video = False
        processor = self.processor
        images = None
        videos = None
        video_kwargs = {}
        pixel_key = None
        grid_key = None
        
        # Handle image/video data
        # This part is similar to DPODataset and SupervisedDataset
        image_folder_path = self.data_args.image_folder
        if not image_folder_path and "image" in sources: # Try to infer image folder if not provided
            # Assuming data_path is a file like /path/to/data/dpo.json
            # and images are in /path/to/data/images/
            base_data_dir = os.path.dirname(self.data_args.data_path)
            inferred_image_folder = os.path.join(base_data_dir, "images")
            if os.path.isdir(inferred_image_folder):
                image_folder_path = inferred_image_folder
            else: # Fallback to current working directory's sepo/data/RLAIF_Sample/DPO_format_for_Qwen/images
                  # This is specific to the user's request.
                image_folder_path = os.path.join(os.getcwd(), "sepo/data/RLAIF_Sample/DPO_format_for_Qwen/images")


        if "image" in sources:
            pixel_key = "pixel_values"
            grid_key = "image_grid_thw"
            image_files = sources["image"]
            if isinstance(image_files, str): image_files = [image_files]
            
            images = []
            for image_file in image_files:
                full_image_path = image_file
                if image_folder_path and not os.path.isabs(image_file) and not image_file.startswith("http"):
                    full_image_path = os.path.join(image_folder_path, image_file)
                if not os.path.exists(full_image_path) and not full_image_path.startswith("http"):
                     print(f"Warning: Image file not found at {full_image_path}, ensure image_folder is correct or images are in dpo.json path.")
                     # Fallback to relative path from dpo.json if image_folder is not set or path is not absolute
                     if not self.data_args.image_folder and isinstance(self.data_args.data_path, str):
                         base_data_dir = os.path.dirname(self.data_args.data_path)
                         potential_path = os.path.join(base_data_dir, image_file)
                         if os.path.exists(potential_path):
                             full_image_path = potential_path
                         else:
                             # Try the user specified path directly if it's absolute
                             if os.path.isabs(image_file) and os.path.exists(image_file):
                                 full_image_path = image_file
                             else: # Final fallback to user's specific image dir
                                full_image_path = os.path.join(os.getcwd(), "sepo/data/RLAIF_Sample/DPO_format_for_Qwen/images", os.path.basename(image_file))


                images.append(get_image_info(full_image_path, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True
            pixel_key = "pixel_values_videos"
            grid_key = "video_grid_thw"
            video_files = sources["video"]
            if isinstance(video_files, str): video_files = [video_files]

            videos = []
            for video_file in video_files:
                full_video_path = video_file
                if image_folder_path and not os.path.isabs(video_file) and not video_file.startswith("http"):
                     full_video_path = os.path.join(image_folder_path, video_file)
                # Similar fallback logic for videos if needed
                current_video_input, current_video_kwargs = get_video_info(full_video_path, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.fps)
                videos.append(current_video_input)
                video_kwargs.update(current_video_kwargs) # Assuming video_kwargs are consistent for all videos in a sample

        # Construct the input sequence from prompt and chosen response
        # This is similar to SupervisedDataset's handling of conversations
        prompt_text = sources["prompt"]
        chosen_text = sources["chosen"]

        # Mimic conversation structure for processor
        # User: prompt, Assistant: chosen
        conversation = [
            {"from": "user", "value": prompt_text},
            {"from": "gpt", "value": chosen_text}
        ]
        
        # Convert to OpenAI format if needed by processor logic (borrowed from SupervisedDataset)
        openai_conversation = llava_to_openai(conversation, is_video=is_video)

        all_input_ids_list = []
        all_labels_list = []
        all_pixel_values_list = []
        all_grid_thw_list = []
        all_second_grid_ts_list = [] # For Qwen2.5 VL video

        if SYSTEM_MESSAGE and len(SYSTEM_MESSAGE) > 0:
            system_message_text = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_ids = processor.tokenizer(system_message_text, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_ids, IGNORE_INDEX)
            all_input_ids_list.append(system_ids.squeeze(0))
            all_labels_list.append(system_labels.squeeze(0))

        # Process user part (prompt)
        user_content = openai_conversation[0]['content']
        user_turn_text = f"{DEFAULT_IM_START_TOKEN}user\n{user_content}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        
        # Process assistant part (chosen)
        assistant_content = openai_conversation[1]['content']
        assistant_turn_text = f"{assistant_content}{DEFAULT_IM_END_TOKEN}\n"

        # Tokenize user part (prompt)
        if images is not None or videos is not None:
            # Multimodal input processing
            proc_kwargs = {"text": [user_turn_text], "images": images, "videos": videos, "padding": False, "do_resize": False, "return_tensors": 'pt'}
            if is_video and "Qwen2.5" in self.model_id: # Specific for Qwen2.5 video
                proc_kwargs.update(video_kwargs)

            inputs_proc = processor(**proc_kwargs)
            prompt_ids = inputs_proc['input_ids'].squeeze(0)
            if pixel_key and grid_key:
                all_pixel_values_list.append(inputs_proc[pixel_key])
                all_grid_thw_list.append(inputs_proc[grid_key])
            if is_video and "Qwen2.5" in self.model_id and "second_per_grid_ts" in inputs_proc:
                 all_second_grid_ts_list.extend(inputs_proc["second_per_grid_ts"])

        else: # Text-only prompt
            prompt_ids = processor.tokenizer(user_turn_text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)

        # Tokenize assistant part (chosen response)
        response_ids = processor.tokenizer(assistant_turn_text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        # Combine prompt and response
        current_input_ids = torch.cat([prompt_ids, response_ids], dim=0)
        # Labels: IGNORE_INDEX for prompt, actual tokens for response
        current_labels = torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), response_ids], dim=0)

        all_input_ids_list.append(current_input_ids)
        all_labels_list.append(current_labels)

        # Concatenate all parts
        final_input_ids = torch.cat(all_input_ids_list, dim=0).to(torch.long)
        final_labels = torch.cat(all_labels_list, dim=0).to(torch.long)
        final_attention_mask = (final_input_ids != processor.tokenizer.pad_token_id).to(torch.long)


        data_dict = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "labels": final_labels, # SEPOTrainer uses inputs['input_ids'] and inputs['attention_mask']
                                    # but standard Trainer expects 'labels' for loss computation if not overridden
                                    # For SEPO, the loss is custom, so 'labels' might not be strictly needed by SEPOTrainer's compute_loss
                                    # but good to keep for compatibility / potential SFT on chosen.
        }

        if len(all_pixel_values_list) > 0 and pixel_key and grid_key:
            data_dict[pixel_key] = torch.cat(all_pixel_values_list, dim=0)
            data_dict[grid_key] = torch.cat(all_grid_thw_list, dim=0)
        
        if len(all_second_grid_ts_list) > 0:
             data_dict["second_grid_ts"] = all_second_grid_ts_list # Note: dpo_trainer uses second_grid_ts, not second_per_grid_ts

        return data_dict


class DataCollatorForSEPODataset(object):
    """Collate examples for SEPO fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [example["input_ids"] for example in examples]
        batch_labels = [example["labels"] for example in examples] # Or however SEPO needs it

        # Pad sequences
        input_ids = pad_sequence(batch_input_ids, padding_side='right', padding_value=self.pad_token_id)
        labels = pad_sequence(batch_labels, padding_side='right', padding_value=IGNORE_INDEX)
        attention_mask = (input_ids != self.pad_token_id)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels, # Keep for consistency, though SEPO loss is custom
        }

        # Handle multimodal data (pixel_values, image_grid_thw, etc.)
        # This logic is similar to DataCollatorForSupervisedDataset / DataCollatorForDPODataset
        if "pixel_values" in examples[0]:
            batch_pixel_values = [example["pixel_values"] for example in examples]
            batch_image_thw = [example["image_grid_thw"] for example in examples]
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if "pixel_values_videos" in examples[0]:
            batch_pixel_video_values = [example["pixel_values_videos"] for example in examples]
            batch_video_thw = [example["video_grid_thw"] for example in examples]
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_video_values, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)
        
        if "second_grid_ts" in examples[0]: # From Qwen2.5 VL video
            batch_second_grid_ts = []
            for ex in examples: # second_grid_ts might be a list of tensors
                if isinstance(ex["second_grid_ts"], list):
                    batch_second_grid_ts.extend(ex["second_grid_ts"])
                else: # if it's a single tensor per example (less likely for list extend)
                    batch_second_grid_ts.append(ex["second_grid_ts"])
            if batch_second_grid_ts:
                 # If elements are tensors, try to stack or cat. If they are already processed, might be fine.
                 # The DPO trainer concatenates them. Let's assume they are tensors that can be concatenated.
                 try:
                    data_dict["second_grid_ts"] = torch.cat(batch_second_grid_ts, dim=0) if all(isinstance(t, torch.Tensor) for t in batch_second_grid_ts) else batch_second_grid_ts
                 except: # Fallback if not tensors or not cat-able
                    data_dict["second_grid_ts"] = batch_second_grid_ts


        return data_dict

def make_sepo_data_module(model_id: str, processor: transformers.ProcessorMixin, data_args: DataArguments) -> Dict:
    """Make dataset and collator for SEPO fine-tuning."""
    sepo_dataset = SEPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSEPODataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sepo_dataset,
                eval_dataset=None, # No eval dataset specified for now
                data_collator=data_collator)

# For dataclass and field if src.train.params is not available
from dataclasses import dataclass, field
from typing import Optional
