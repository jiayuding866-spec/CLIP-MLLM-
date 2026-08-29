# 从相似度矩阵到视觉问答：CLIP 与 MLLM 的能力边界

这是一份课堂实战 Notebook。它不把 CLIP、检索、线性探针和视觉问答拆成四段互不相关的 API 演示，而是反复问同一个问题：

**一个只会算“图和文有多像”的模型，到底能做什么？什么时候必须换成会写句子的多模态大模型？**

主线是 **CLIP**。SigLIP 2 只做对照，用来看训练目标和分数含义差在哪里。最后用 **Qwen3-VL-2B-Instruct** 做开放式视觉问答，对比两种输出形式。

---

## 仓库里有什么

| 文件 | 说明 |
|---|---|
| `从相似度矩阵到视觉问答_CLIP与MLLM能力边界_课堂实战.ipynb` | 可运行的课堂 Notebook（无输出，适合自己从头跑） |
| `从相似度矩阵到视觉问答_CLIP与MLLM能力边界_课堂实战.executed.ipynb` | 同内容的一次完整运行结果，方便先看图和数字 |
| `requirements.txt` | Python 依赖（不包含 PyTorch，请按自己的 CUDA 版本安装） |

模型和数据集不会进 Git。第一次运行时会下载 CLIP、SigLIP 2、Qwen3-VL、CIFAR-100 和 Flickr1K。

---

## 你会做哪些实验

1. **拆开 CLIP**  
   不只调用 pipeline。看图像编码器、文本编码器、投影层和温度参数各自干什么，并用 hook 核对中间张量形状。

2. **零样本分类**  
   把 CIFAR-100 的类别写成英文 prompt，用相似度矩阵当临时分类器。比较单模板和 prompt ensemble。

3. **图文双向检索**  
   在 Flickr1K 上做 Image→Text 和 Text→Image 的 Recall@K。注意每张图有 5 条描述，正确答案不只在矩阵对角线上。

4. **冻结 CLIP，训练 Linear Probe**  
   视觉塔不再更新。只在缓存好的图像特征上训练一层线性分类器，并和零样本对比。

5. **Qwen3-VL 视觉问答**  
   同一张图先让 CLIP 在固定候选里打分，再让 Qwen3-VL 自由生成描述和空间关系。看 chat template、视觉 token、prefill 和 `generate()` 分别做了什么。

课堂建议 3～4 小时。Notebook 顶部有开关：`QUICK_MODE=True` 时只用小子集，适合当堂演示。

---

## 怎么运行

1. 建议使用 GPU。CLIP 三个实验 8GB 显存较舒服；Qwen3-VL-2B 更稳妥的是 12GB 以上。CPU 能加载 CLIP 的小子集，但不适合当堂跑大模型。

2. 先按 [pytorch.org](https://pytorch.org) 安装匹配本机 CUDA 的 `torch` 和 `torchvision`，再安装其余依赖：

```bash
pip install -r requirements.txt
```

请使用 **Transformers 4.57.x**。5.x 会改变 CLIP `get_image_features()` 的返回类型，这份 Notebook 会报错。`datasets` 请留在 3.x：Flickr1K 仍依赖数据集脚本，4/5 会加载失败。

3. 用 Jupyter / VS Code / Cursor 打开 `.ipynb`，按单元格运行。环境已经配好时，可以跳过第一个 `%pip install` 单元格。

4. 如果访问 `huggingface.co` 很慢或不通（常见于国内机器），在运行前设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
```

---

## 一次实际运行的参考数字

下面是在 RTX 4080 SUPER、`QUICK_MODE=True` 下的一次结果，只用来建立数量级直觉。换机器、换子集、换随机种子都会有差异。

| 实验 | 设定 | 结果 |
|---|---|---|
| CLIP 零样本 | CIFAR-100，1000 张测试图 | 单模板 Top-1 66.0%；prompt ensemble Top-1 66.4% |
| 图文检索 | Flickr1K 前 100 张图 | Image→Text R@1 97%；Text→Image R@1 84.4% |
| Linear Probe | 5000 训练 / 1000 测试 | sklearn 73.4%；PyTorch 线性头 73.0% |
| CLIP 问海滩图 | 4 个固定候选 | “人与狗在海滩”得到全部概率质量 |
| Qwen3-VL | 同一张图开放问答 | 生成了海滩、击掌、牵引绳等描述 |

检索数字在 100 张图上会偏高。放到完整 1000 张时，Recall@1 通常会下降，因为干扰项变多了。

---

## 读结果时记住的几件事

- CLIP 的 softmax 只是“在当前候选集合里怎么分”，不是校准过的真实把握。候选一变，数字就变。漏掉正确类别时，它仍会从错误选项里挑一个最高分，而不会拒答。
- SigLIP 的 sigmoid 分数和 CLIP 的 softmax 不能直接比大小。一个是独立的 pairwise 匹配分，一个是行内相对竞争。
- Linear Probe 高于零样本，说明视觉特征里已经有不少类别信息；差距更常出在文本提示有没有把决策边界用满，而不一定是 encoder “不够强”。
- Qwen 写得流畅，不等于解释一定真实。生成模型可以编细节。

---

## 代码与文档依据

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
- [Transformers SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) / [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Flickr1K 检索测试集](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval)
