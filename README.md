# LC-CLAP

LC-CLAP is a Leq-conditioned audio-text learning framework for ten audio events. Task definitions and default training parameters are available in [configs/roadside_events_public.json](configs/roadside_events_public.json).

Model sources:

- B0: PANNs CNN14, based on the [original implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- B1-B4: HTSAT-tiny from [Hugging Face](https://huggingface.co/laion/clap-htsat-unfused)
- B2-B4 text encoder: [RoBERTa-base](https://huggingface.co/roberta-base)
- Model interface: [Transformers CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap)

Run training, evaluation, and single-file inference with `train.py`, `evaluate.py`, and `retrieve.py`, respectively. For B2-B4, specify a Hugging Face model ID or a local model directory from the command line.

This project is released under the MIT License.
