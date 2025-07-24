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

Here, we removed the visual components of qwen2.5-vl and merged all LoRA adapters back into the base language model. This results in three task-specific v4 models with 3.09B parameters, downsized from the original jina-embeddings-v4 3.75B parameters:

| HuggingFace Repo | Task |
|---|---|
| [`jinaai/jina-embeddings-v4-text-retrieval-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF) | Text retrieval |
| [`jinaai/jina-embeddings-v4-text-code-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-code-GGUF) | Code retrieval |
| [`jinaai/jina-embeddings-v4-text-matching-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-matching-GGUF) | Sentence similarity |

All models above provide F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M quantizations. More quantizations such as Unsloth-like dynamic quantizations are on the way.

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
