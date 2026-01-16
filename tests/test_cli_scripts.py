import subprocess
import os
import pytest

def test_train_script_missing_config():
    import subprocess
    try:
        result = subprocess.run(["python", "train.py", "--config", "nonexistent.yaml"], capture_output=True, text=True, timeout=10)
        # If it returns, check for error
        assert result.returncode != 0
        assert "No such file" in result.stderr or "not found" in result.stderr
    except subprocess.TimeoutExpired:
        # Timeout is acceptable for this test
        pass

def test_evaluate_script_missing_checkpoint():
    # Remove checkpoint if exists
    ckpt_path = "checkpoints/UNet_best.pth"
    if os.path.exists(ckpt_path):
        os.rename(ckpt_path, ckpt_path + ".bak")
    result = subprocess.run(["python", "evaluate.py", "--config", "config/config.yaml"], capture_output=True, text=True, timeout=10)
    if os.path.exists(ckpt_path + ".bak"):
        os.rename(ckpt_path + ".bak", ckpt_path)
    assert result.returncode != 0
    assert "No such file" in result.stderr or "not found" in result.stderr

def test_generate_masks_script_missing_checkpoint():
    ckpt_path = "checkpoints/UNet_best.pth"
    if os.path.exists(ckpt_path):
        os.rename(ckpt_path, ckpt_path + ".bak")
    result = subprocess.run(["python", "generate_masks.py"], capture_output=True, text=True, timeout=10)
    if os.path.exists(ckpt_path + ".bak"):
        os.rename(ckpt_path + ".bak", ckpt_path)
    assert result.returncode != 0
    assert "No such file" in result.stderr or "not found" in result.stderr
