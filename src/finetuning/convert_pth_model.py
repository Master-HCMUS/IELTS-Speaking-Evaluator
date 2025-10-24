"""
Utility script to convert and use PyTorch .pth Whisper models.

This script helps integrate PyTorch .pth format Whisper models with the existing system.
"""

import torch
import os
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_pth_whisper_model(pth_path: str, base_model: str = "openai/whisper-tiny"):
    """
    Load a fine-tuned Whisper model from a .pth file.
    
    Args:
        pth_path: Path to the .pth file
        base_model: Base Whisper model to use as foundation
        
    Returns:
        tuple: (model, processor)
    """
    print(f"Loading PyTorch model from: {pth_path}")
    
    # Load the base model and processor
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    processor = WhisperProcessor.from_pretrained(base_model)
    
    # Load the fine-tuned weights
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Model loaded successfully from {pth_path}")
    return model, processor


def convert_pth_to_transformers_format(pth_path: str, output_dir: str, base_model: str = "openai/whisper-tiny"):
    """
    Convert a .pth model to Transformers format for easier integration.
    
    Args:
        pth_path: Path to the .pth file
        output_dir: Directory to save the converted model
        base_model: Base Whisper model to use
    """
    print(f"Converting {pth_path} to Transformers format...")
    
    # Load the model
    model, processor = load_pth_whisper_model(pth_path, base_model)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in Transformers format
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    # Save additional metadata
    metadata = {
        "source_file": str(pth_path),
        "base_model": base_model,
        "conversion_method": "pth_to_transformers",
        "model_type": "fine-tuned-whisper"
    }
    
    import json
    with open(output_path / "conversion_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model converted and saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch .pth Whisper model to Transformers format")
    parser.add_argument("--pth-path", required=True, help="Path to the .pth file")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted model")
    parser.add_argument("--base-model", default="openai/whisper-tiny", help="Base Whisper model")
    
    args = parser.parse_args()
    
    convert_pth_to_transformers_format(args.pth_path, args.output_dir, args.base_model)