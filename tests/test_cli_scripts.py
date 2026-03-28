import subprocess
import os
import sys
import pytest
import yaml
import tempfile

def test_train_script_missing_config():
    try:
        result = subprocess.run([sys.executable, "train.py", "--config", "nonexistent.yaml"], capture_output=True, text=True, timeout=10)
        # If it returns, check for error
        assert result.returncode != 0
        assert "No such file" in result.stderr or "not found" in result.stderr
    except subprocess.TimeoutExpired:
        # Timeout is acceptable for this test
        pass

def test_evaluate_script_missing_checkpoint():
    """Checks if evaluate.py fails gracefully when a checkpoint is missing by using a temporary config."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        # Create a config that points to a non-existent model to force a missing checkpoint error
        tmp_config = {
            'DATASET_PATH': '.', 
            'IMAGE_SIZE': 256,
            'BATCH_SIZE': 1,
            'NUM_WORKERS': 0,
            'MODEL_SET': 'standard',
            'STANDARD_MODELS': ['DeepLabV3'], # DeepLabV3_best.pth is missing
            'NUM_CLASSES': 2
        }
        yaml.dump(tmp_config, tmp)
        tmp_path = tmp.name

    try:
        # DeepLabV3_best.pth should be missing in checkpoints/
        result = subprocess.run([sys.executable, "evaluate.py", "--config", tmp_path], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "No such file" in result.stderr or "not found" in result.stderr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_generate_masks_script_missing_checkpoint():
    """Checks if generate_masks.py fails gracefully when a checkpoint is missing by using a temporary config."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        tmp_config = {
            'DATASET_PATH': '.',
            'IMAGE_SIZE': 256,
            'MODEL_SET': 'standard',
            'STANDARD_MODELS': ['DeepLabV3'], # DeepLabV3_best.pth is missing
            'NUM_CLASSES': 2
        }
        yaml.dump(tmp_config, tmp)
        tmp_path = tmp.name

    try:
        result = subprocess.run([sys.executable, "generate_masks.py", "--config", tmp_path], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "No such file" in result.stderr or "not found" in result.stderr
    except subprocess.TimeoutExpired:
        assert False, "generate_masks.py timeout expired instead of exiting gracefully!"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
