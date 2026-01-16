import torch
import random
import numpy as np

def test_global_seed_reproducibility():
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Generate random tensors and arrays
    arr1 = np.random.rand(5)
    t1 = torch.randn(5)
    random_val1 = random.random()
    # Reset seeds and generate again
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    arr2 = np.random.rand(5)
    t2 = torch.randn(5)
    random_val2 = random.random()
    assert np.allclose(arr1, arr2)
    assert torch.allclose(t1, t2)
    assert random_val1 == random_val2

def test_mixed_precision_context():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Conv2d(3, 2, 1).to(device)
    x = torch.randn(2, 3, 8, 8, device=device)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler is not None:
        with torch.amp.autocast('cuda'):
            y = model(x)
            loss = y.sum()
        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters()))
        scaler.update()
    else:
        y = model(x)
        loss = y.sum()
        loss.backward()
    assert y.shape[0] == 2
    assert y.shape[1] == 2
