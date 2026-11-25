import os

# Apple Silicon: op nào MPS chưa hỗ trợ thì chạy CPU thay vì crash. Phải set trước khi import torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
