---
license: cc-by-nc-4.0
base_model:
- jinaai/jina-embeddings-v4
base_model_relation: quantized
---

# jina-embeddings-v4-gguf

A collection of GGUF and quantizations for [`jina-embeddings-v4`](https://huggingface.co/jinaai/jina-embeddings-v4).

## Overview

`jina-embeddings-v4` is a cutting-edge universal embedding model [for multimodal multilingual retrieval](https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval). It's based on `qwen2.5-vl-3b-instruct` with three LoRA adapters: `retrieval` (optimized for retrieval tasks), `text-matching` (optimized for sentence similarity tasks), and `code` (optimized for code retrieval tasks). It is also heavily trained for visual document retrieval and late-interaction style multi-vector output.

## Text-Only Task-Specific Models

We removed the visual components of `qwen2.5-vl` and merged all LoRA adapters back into the base language model. This results in three task-specific v4 models with 3.09B parameters, downsized from the original jina-embeddings-v4 3.75B parameters:

| HuggingFace Repo | Task |
|---|---|
| [`jinaai/jina-embeddings-v4-text-retrieval-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF) | Text retrieval |
| [`jinaai/jina-embeddings-v4-text-code-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-code-GGUF) | Code retrieval |
| [`jinaai/jina-embeddings-v4-text-matching-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-matching-GGUF) | Sentence similarity |

All models above provide F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M and dynamic quantizations such as IQ1_S, IQ2_XXS.

### Limitations vs original v4 model
- They can not handle image input.
- They can not output multi-vector embeddings.
- You must add `Query: ` or `Passage: ` in front of the input. [Check this table for the details](#consistency-wrt-automodelfrom_pretrained).
 
## Multimodal Task-Specific Models

TBA

## Get Embeddings

First [install llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

Run `llama-server` to host the embedding model as OpenAI API compatible HTTP server. As an example for using `text-matching` with `F16`, you can do:

```bash
llama-server -hf jinaai/jina-embeddings-v4-text-matching-GGUF:F16 --embedding --pooling mean -ub 8192
```

Remarks:
- `--pooling mean` is required as v4 is mean-pooling embeddings.
- setting `--pooling none` is *not* as same as the multi-vector embeddings of v4. The original v4 has a trained MLP on top of the last hidden states to output multi-vector embeddings, each has 128-dim. In GGUF, this MLP was chopped off.

Client:

```bash
curl -X POST "http://127.0.0.1:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Query: A beautiful sunset over the beach",
      "Query: Un beau coucher de soleil sur la plage",
      "Query: 海滩上美丽的日落",
      "Query: 浜辺に沈む美しい夕日"
    ]
  }'
```

Note: When using `retrieval` and `code` models, add `Query: ` or `Passage:` in front of your input, like this:

```bash
curl -X POST "http://127.0.0.1:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Query: A beautiful sunset over the beach",
      "Query: Un beau coucher de soleil sur la plage",
      "Passage: 海滩上美丽的日落",
      "Passage: 浜辺に沈む美しい夕日"
    ]
  }'
```


You can also use `llama-embedding` for one-shot embedding:

```bash
llama-embedding -hf jinaai/jina-embeddings-v4-text-matching-GGUF:F16 --pooling mean -p "Query: jina is awesome" --embd-output-format json  2>/dev/null
```

## Remarks

### Consistency wrt. `AutoModel.from_pretrained`

To get fully consistent results as if you were [using `AutoModel.from_pretrained("jinaai/jina-embeddings-v4")...`](https://huggingface.co/jinaai/jina-embeddings-v4#usage), you need to be **very careful** about the prefixes and manually add them to your GGUF model inputs. Here's a reference table:

| Input Type | Task | `prompt_name` (Role) | Actual Input Processed by Model |
|------------|------|-------------|-------------------------------|
| **Text** | `retrieval` | `query` (default) | `Query: {original_text}` |
| **Text** | `retrieval` | `passage` | `Passage: {original_text}` |
| **Text** | `text-matching` | `query` (default) | `Query: {original_text}` |
| **Text** | `text-matching` | `passage` | `Query: {original_text}` ⚠️ |
| **Text** | `code` | `query` (default) | `Query: {original_text}` |
| **Text** | `code` | `passage` | `Passage: {original_text}` |
| **Image** | Any task | N/A | `<\|im_start\|>user\n<\|vision_start\|>\<\|image_pad\|>\<\|vision_end\|>Describe the image.\<\|im_end\|>` |


To some users, ⚠️ indicates a somewhat surprising behavior where `prompt_name='passage'` gets overridden to `"Query: "` when using `text-matching` in the original `AutoModel.from_pretrained("jinaai/jina-embeddings-v4")....` However, this is reasonable since `text-matching` is a sentence similarity task with no left/right roles—the inputs are symmetric.


### Matryoshka embeddings

Note, v4 is trained with Matryoshka embeddings, and converting to GGUF doesn't break the Matryoshka feature. Let's say you get embeddings with shape `NxD` - you can simply use `embeddings[:, :truncate_dim]` to get smaller truncated embeddings. Note that not every dimension is trained though. For v4, you can set `truncate_dim` to any of these values: `[128, 256, 512, 1024, 2048]`.

### Quantizations

We use [`llama-quantize`](./quantize.sh) with `imatrix` to quantize models from float16. `imatrix` is generated by `llama-imatrix -m jina-embeddings-v4-text-retrieval-F16.gguf -f calibration_data_v5_rc.txt -ngl 99 --no-ppl -o imatrix-retrieval-512.dat`. `calibration_data_v5_rc.txt` can be found [here](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/) and is recommended by Unsloth docs.


Here's the speed and quality evaluation on two nano benchmarks. The higher the better. `IQ3_XXS` seems to be a good balance between size and speed.

![](https://raw.githubusercontent.com/jina-ai/jina-embeddings-v4-gguf/refs/heads/main/gguf-v4-on-l4.svg)
![](https://raw.githubusercontent.com/jina-ai/jina-embeddings-v4-gguf/refs/heads/main/NanoHotpotQA.svg)
![](https://raw.githubusercontent.com/jina-ai/jina-embeddings-v4-gguf/refs/heads/main/NanoFiQA2018.svg)

#### NDCG@5
| Quantization Type | NanoHotpotQA | NanoFiQA2018 | Δ to v3 (HotpotQA) | Δ to v4 (HotpotQA) | Δ to v3 (FiQA2018) | Δ to v4 (FiQA2018) |
|------------------|--------------|--------------|-------------------|-------------------|-------------------|-------------------|
| IQ1_S | 0.6369 | 0.3178 | -14% | -20% | -38% | -43% |
| IQ1_M | 0.6316 | 0.3313 | -15% | -21% | -36% | -41% |
| IQ2_XXS | 0.7236 | 0.4582 | -2% | -9% | -11% | -18% |
| IQ2_M | 0.7427 | 0.5869 | +0% | -7% | +14% | +5% |
| Q2_K | 0.7683 | 0.5744 | +4% | -4% | +12% | +3% |
| IQ3_XXS | 0.7780 | 0.5991 | +5% | -2% | +16% | +8% |
| IQ3_XS | 0.7727 | 0.5615 | +5% | -3% | +9% | +1% |
| IQ3_S | 0.8002 | 0.5505 | +8% | +0% | +7% | -1% |
| IQ3_M | 0.8106 | 0.5387 | +10% | +2% | +5% | -3% |
| Q3_K_M | 0.7567 | 0.5267 | +2% | -5% | +2% | -5% |
| IQ4_NL | 0.7930 | 0.5598 | +7% | -1% | +9% | +0% |
| IQ4_XS | 0.7979 | 0.5627 | +8% | +0% | +9% | +1% |
| Q4_K_M | 0.8029 | 0.5569 | +9% | +1% | +8% | +0% |
| Q5_K_S | 0.7969 | 0.5581 | +8% | +0% | +8% | +0% |
| Q5_K_M | 0.7927 | 0.5601 | +7% | -1% | +9% | +1% |
| Q6_K | 0.7951 | 0.5636 | +8% | +0% | +10% | +1% |
| Q8_0 | 0.7938 | 0.5687 | +7% | +0% | +11% | +2% |
| F16 | 0.7940 | 0.5610 | +7% | +0% | +9% | +1% |
| jinaai-jina-embeddings-v3 | 0.7393 | 0.5144 | +0% | -7% | +0% | -8% |
| jinaai-jina-embeddings-v4 | 0.7977 | 0.5571 | +8% | +0% | +8% | +0% |

#### Tokens per Second
| Quantization Type | NanoHotpotQA | NanoFiQA2018 | Δ to F16 (HotpotQA) | Δ to F16 (FiQA2018) |
|------------------|--------------|--------------|--------------------|--------------------|
| IQ1_S | 1608 | 1618 | +53% | +49% |
| IQ1_M | 1553 | 1563 | +48% | +44% |
| IQ2_XXS | 1600 | 1612 | +52% | +49% |
| IQ2_M | 1529 | 1534 | +46% | +42% |
| Q2_K | 1459 | 1471 | +39% | +36% |
| IQ3_XXS | 1552 | 1487 | +48% | +37% |
| IQ3_XS | 1529 | 1526 | +46% | +41% |
| IQ3_S | 1520 | 1516 | +45% | +40% |
| IQ3_M | 1507 | 1511 | +44% | +40% |
| Q3_K_M | 1475 | 1487 | +40% | +37% |
| IQ4_NL | 1464 | 1469 | +39% | +36% |
| IQ4_XS | 1478 | 1487 | +41% | +37% |
| Q4_K_M | 1454 | 1458 | +38% | +35% |
| Q5_K_S | 1419 | 1429 | +35% | +32% |
| Q5_K_M | 1404 | 1433 | +34% | +32% |
| Q6_K | 1356 | 1382 | +29% | +28% |
| Q8_0 | 1304 | 1334 | +24% | +23% |
| F16 | 1050 | 1083 | +0% | +0% |
