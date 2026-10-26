# CASI: Category-Shared Industrial Anomaly Generation with Selective Interaction

## 环境

在 Python 3.10 或 3.11 环境中，先准备适配显卡的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

模型使用本地 Stable Diffusion v1.5 目录，至少含 `tokenizer/`、`text_encoder/`、`unet/`、`vae/`、`scheduler/` 和 `model_index.json`。代码设置 `local_files_only=True`，不会自动下载模型。

## MVTec AD：训练与生成

以下路径替换成你实际的模型和数据目录；命令在 `CASI/` 目录运行。每条命令处理一个类别，默认自动收集该类别的全部缺陷。

```bash
python train_codd.py \
  --model_path /path/to/stable-diffusion-v1-5 \
  --data_root /path/to/mvtec \
  --dataset mvtec --category bottle \
  --output_dir checkpoints/mvtec/bottle/codd

python train_dgsi.py \
  --model_path /path/to/stable-diffusion-v1-5 \
  --codd_path checkpoints/mvtec/bottle/codd \
  --data_root /path/to/mvtec \
  --dataset mvtec --category bottle \
  --mixed_precision fp16 \
  --output_dir checkpoints/mvtec/bottle/dgsi

python generate.py \
  --model_path /path/to/stable-diffusion-v1-5 \
  --codd_path checkpoints/mvtec/bottle/codd \
  --dgsi_path checkpoints/mvtec/bottle/dgsi \
  --category bottle \
  --num_images 1000 --output_dir generated/mvtec
```

