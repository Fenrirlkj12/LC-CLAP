# 工作流

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据清单

UTF-8 JSONL，每行至少包含 `audio_path` 和 `label`。B3、B4、B5、B7 需要 `leq_value`，可选 `split` 用于 split 内 shuffle。

```json
{"audio_path":"audio/clip_001.wav","label":"Light vehicle pass-by","leq_value":69.4,"split":"train"}
```

音频会转换为单声道、重采样到 48 kHz，并裁剪或补齐到 10 秒。

## 训练

```bash
python train.py \
  --manifest data/train.jsonl \
  --task-config configs/roadside_events_public.json \
  --experiment B4 \
  --pretrained-audio laion/clap-htsat-unfused \
  --pretrained-text roberta-base \
  --output-dir runs/B4_seed3408
```

`B0`、`B1` 使用音频分类头；`B2` 为无 Leq 提示的音频文本对比学习；`B3` 使用硬 Leq 桶提示；`B4` 使用五档 Leq 的 Gaussian 软权重。默认 epochs、batch size、learning rate、weight decay、workers、seed 等参数位于任务配置的 `training_defaults`，也可以用命令行参数覆盖。

B0 可通过 `--pretrained-audio` 指定兼容 PANNs checkpoint；B1 可指定 Hugging Face HTSAT 模型 ID 或本地 Transformers 目录。B2-B4 需要同时指定 `--pretrained-audio` 和 `--pretrained-text`。

B5 和 B7 是 B3 的 Leq shuffle 控制：

```bash
python train.py --manifest data/train.jsonl \
  --task-config configs/roadside_events_public.json \
  --experiment B5_shuffle_control \
  --pretrained-audio laion/clap-htsat-unfused \
  --pretrained-text roberta-base \
  --output-dir runs/B5_seed3408
```

## 评估

```bash
python evaluate.py \
  --checkpoint runs/B4_seed3408/last.pt \
  --manifest data/validation.jsonl \
  --task-config configs/roadside_events_public.json \
  --output outputs/B4_seed3408_validation.json
```

输出 Top-1 accuracy、Macro-F1 和逐类指标。多次运行和结果汇总可由使用者按需要编写脚本完成。

## 单文件推理

```bash
python retrieve.py \
  --checkpoint runs/B4_seed3408/last.pt \
  --audio data/example.wav \
  --task-config configs/roadside_events_public.json \
  --leq 71.2 \
  --top-k 5
```

B3、B4 推理需要 `--leq`；B2、B0、B1 不需要。
