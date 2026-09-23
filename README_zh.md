# 从相似度分数到视觉问答

[English](README.md) · [中文](README_zh.md)

这份仓库记录我做的一组对照实验：**一个只会算“图和文有多像”的模型能做什么，什么时候必须换成会写句子的多模态大模型。**

主线是 **CLIP**（`openai/clip-vit-base-patch32`）。我拆开图像编码器、文本编码器、投影层和温度，再在同一套代码上跑四组实验：

1. CIFAR-100 上的 prompt 零样本分类  
2. Flickr1K 上的图文双向检索  
3. 冻结 CLIP 后的 Linear Probe  
4. 同一张图上，CLIP 只能在固定候选里打分，**Qwen3-VL-2B-Instruct** 则开放式问答  

**SigLIP 2** 只做对照：看训练目标和分数含义差在哪里，而不是把 CLIP 实验再做一遍。

完整数字、复现方式和读结果时的注意点见 [English README](README.md)。执行过的全量结果在 [`clip_mllm_boundary.executed.ipynb`](clip_mllm_boundary.executed.ipynb)。
