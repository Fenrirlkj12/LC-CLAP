# LC-CLAP Demo

十类音频事件的 Leq 条件音频文本学习示例。任务定义和训练默认参数见 [configs/roadside_events_public.json](configs/roadside_events_public.json)

模型来源：

- B0：PANNs CNN14，[原始实现](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- B1-B4：HTSAT-tiny，[Hugging Face 模型](https://huggingface.co/laion/clap-htsat-unfused)
- B2-B4 文本编码器：[RoBERTa-base](https://huggingface.co/roberta-base)
- 模型接口：[Transformers CLAP 文档](https://huggingface.co/docs/transformers/model_doc/clap)

训练、评估和单文件推理分别由 `train.py`、`evaluate.py` 和 `retrieve.py` 启动。运行 B2-B4 时通过命令行指定 Hugging Face 模型 ID 或本地模型目录。

项目使用 MIT License。
