import os
import json
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
import pytest
from amber_generate import load_model_and_processor, process_images


@pytest.fixture
def mock_model_path():
    return "/mock/model/path"


@pytest.fixture
def mock_image_folder(tmp_path):
    # 创建临时图片文件
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    for i in range(3):
        img = Image.new('RGB', (100, 100))
        img_path = image_folder / f"AMBER_{i}.jpg"
        img.save(img_path)
    return image_folder


@patch('amber_generate.Qwen2_5_VLForConditionalGeneration.from_pretrained')
@patch('amber_generate.AutoProcessor.from_pretrained')
def test_load_model_and_processor(mock_processor, mock_model, mock_model_path):
    """测试模型和处理器加载功能"""
    mock_model.return_value = MagicMock()
    mock_processor.return_value = MagicMock()
    model, processor = load_model_and_processor(mock_model_path)
    assert model is not None
    assert processor is not None
    mock_model.assert_called_once()
    mock_processor.assert_called_once()


@patch('amber_generate.Qwen2_5_VLForConditionalGeneration.from_pretrained')
@patch('amber_generate.AutoProcessor.from_pretrained')
@patch('amber_generate.process_vision_info')
def test_process_images(mock_process_vision_info, mock_processor, mock_model, mock_model_path, mock_image_folder):
    """测试图片处理功能"""
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_processor_instance = MagicMock()
    mock_processor.return_value = mock_processor_instance
    mock_process_vision_info.return_value = (MagicMock(), MagicMock())

    model, processor = load_model_and_processor(mock_model_path)
    results = process_images(model, processor, str(mock_image_folder))

    assert len(results) == 3
    for result in results:
        assert "id" in result
        assert "response" in result


@patch('amber_generate.Qwen2_5_VLForConditionalGeneration.from_pretrained')
@patch('amber_generate.AutoProcessor.from_pretrained')
@patch('amber_generate.process_vision_info')
@patch('sepo.sepo_trainer.LowRankNoiseInjection')
def test_noise_injection_effect(mock_noise_injector, mock_process_vision_info, mock_processor, mock_model,
                                mock_model_path, mock_image_folder):
    """测试噪声注入对模型输出的影响"""
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_processor_instance = MagicMock()
    mock_processor.return_value = mock_processor_instance
    mock_process_vision_info.return_value = (MagicMock(), MagicMock())
    mock_noise_injector_instance = MagicMock()
    mock_noise_injector.return_value = mock_noise_injector_instance

    model, processor = load_model_and_processor(mock_model_path)

    # 无噪声时的输出
    results_without_noise = process_images(model, processor, str(mock_image_folder))

    # 注入噪声后的输出
    noisy_model = mock_noise_injector(model)
    results_with_noise = process_images(noisy_model, processor, str(mock_image_folder))

    assert results_without_noise != results_with_noise


