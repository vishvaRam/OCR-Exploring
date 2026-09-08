from huggingface_hub import snapshot_download

repo_id = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
save_dir = "./Models/PP-OCRv6_medium_det_onnx"

model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,  # Saves real files instead of symlinks
)

print(f"Model successfully saved to: {model_path}")