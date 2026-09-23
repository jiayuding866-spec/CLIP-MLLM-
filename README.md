# From Similarity Scores to Visual QA

[English](README.md) · [中文](README_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CLIP](https://img.shields.io/badge/CLIP-Radford%20et%20al.-b31b1b.svg)](https://arxiv.org/abs/2103.00020)

This repository is an independent study of **what a similarity model can do with language, and where it has to give way to a model that writes sentences**.

The spine is **CLIP** (`openai/clip-vit-base-patch32`). I inspect the image encoder, text encoder, projection, and temperature, then run four experiments on the same stack:

1. Prompt-based zero-shot classification on CIFAR-100
2. Bidirectional image–text retrieval on Flickr1K
3. A frozen linear probe on CLIP image features
4. Open-ended visual QA with **Qwen3-VL-2B-Instruct**, on the same image CLIP can only score against a fixed candidate list

**SigLIP 2** is a control, not a second copy of the CLIP experiments. I use it to compare training objectives and what the scores mean (softmax over a candidate set vs independent sigmoid matching).

I did not treat this as four unrelated API demos. The question throughout is: *if a model can only say how similar an image is to a piece of text, what can it still do — and when must you switch to generation?*

---

## What I implemented

| Experiment | What I wrote | What it measures |
|---|---|---|
| Open CLIP | Forward hooks on embeddings / attention / projections; a manual rebuild of `normalize → dot → temperature → softmax` | Whether I am actually using CLIP’s similarity, not a black-box pipeline |
| Zero-shot | English class prompts; single template vs prompt ensemble | How much the **wording of the label** moves Top-1 / Top-5 |
| Retrieval | Image→Text and Text→Image Recall@K on Flickr1K (5 captions per image, so the diagonal is not the only correct cell) | Ranking, not generation |
| Linear probe | Frozen CLIP; sklearn and a 10-epoch PyTorch linear head | How much class signal is already in the visual features |
| Visual QA | Same beach photo: CLIP picks among 4 captions; Qwen3-VL describes and answers follow-ups | Closed-set matching vs free-form language |

Models and datasets are downloaded at runtime. They are not in Git.

---

## Full-run numbers

Hardware: RTX 4080 SUPER. Setting: `QUICK_MODE=False` in [`clip_mllm_boundary.executed.ipynb`](clip_mllm_boundary.executed.ipynb). Another GPU or software stack will move the digits.

| Experiment | Protocol | Result |
|---|---|---|
| CLIP zero-shot | CIFAR-100, all 10,000 test images | Single template Top-1 **64.47%** / Top-5 88.38%. Prompt ensemble Top-1 **65.05%** / Top-5 88.76% |
| Linear probe | 50,000 train / 10,000 test, CLIP frozen | sklearn **80.07%**; PyTorch linear head, 10 epochs **79.79%** |
| Retrieval | Flickr1K, 1,000 images / 5,000 captions | Image→Text R@1 **79.40%**, R@5 95.00%, R@10 98.10%. Text→Image R@1 **58.84%**, R@5 83.46%, R@10 90.04% |
| CLIP on the beach photo | 4 fixed captions | All probability mass on “a person and a dog on a beach” |
| Qwen3-VL | Same photo, open description + 4 follow-ups | Names the scene, spatial relation, animal count, indoor/outdoor evidence — none of which CLIP can say unless the caption is already in the list |

A smoke test (`QUICK_MODE=True`) is not the result. On Flickr, going from 100 to 1,000 images dropped Text→Image R@1 from about 84% to about 59%. That is more distractors, not a broken model.

---

## How I read the numbers

- **Prompting is part of the model.** Ensemble vs one template is a small but real lift. The linear probe sitting ~15 points above zero-shot is the larger signal: the visual features already contain a lot of class information. A lot of the remaining zero-shot error is how the class is written in English, not “the encoder is weak.”
- **CLIP softmax is competition inside the current candidate set**, not a calibrated probability. Change the labels, the numbers move. Leave the true class out and CLIP still picks a winner from the wrong list. It does not abstain.
- **SigLIP sigmoid scores are not the same kind of number as CLIP softmax.** One is pairwise matching; the other is relative ranking in a row. Do not compare them as if they were the same confidence.
- **Retrieval is not QA.** CLIP is cheap once vectors are cached. Qwen3-VL can answer a question that was never in the caption list, and it can also invent details that look fluent.

A practical split: CLIP / SigLIP to retrieve a shortlist, then an MLLM to describe, re-rank, or answer.

---

## Reproduce

Use **Transformers 4.57.x**. 5.x changes the return type of CLIP `get_image_features()`. Keep `datasets` on 3.x: Flickr1K still depends on a dataset script.

GPU: CLIP is comfortable at ~8 GB. Qwen3-VL-2B is happier at 12 GB+. CPU can load a CLIP subset; it is not a good way to run the MLLM.

```bash
git clone https://github.com/jiayuding866-spec/CLIP-MLLM-.git
cd CLIP-MLLM-

# install torch / torchvision for your CUDA build from https://pytorch.org first
pip install -r requirements.txt
```

Open [`clip_mllm_boundary.ipynb`](clip_mllm_boundary.ipynb) and run all cells. For a smoke test leave `QUICK_MODE = True`. For the table above, set `QUICK_MODE = False` (this is already the setting in the executed notebook).

If Hugging Face is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
```

| File | Role |
|---|---|
| [`clip_mllm_boundary.ipynb`](clip_mllm_boundary.ipynb) | Runnable experiment notebook (no outputs) |
| [`clip_mllm_boundary.executed.ipynb`](clip_mllm_boundary.executed.ipynb) | The full run that produced the table |
| [`requirements.txt`](requirements.txt) | Python deps (not PyTorch) |

---

## References

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
- [Transformers SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) / [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Flickr1K retrieval split](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval)

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={ICML},
  year={2021}
}
```

Code is MIT ([LICENSE](LICENSE)). CLIP, SigLIP, Qwen3-VL, CIFAR-100, and Flickr1K follow their own licences.

Author: [jiayuding866-spec](https://github.com/jiayuding866-spec).
