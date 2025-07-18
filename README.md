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
| [`jina-embeddings-v4-text-retrieval-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF) | Text retrieval |
| [`jina-embeddings-v4-text-code-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-code-GGUF) | Code retrieval |
| [`jina-embeddings-v4-text-matching-GGUF`](https://huggingface.co/jinaai/jina-embeddings-v4-text-matching-GGUF) | Sentence similarity |

All models above provide F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M quantizations. 

### Limitations
- They can not handle image input.
- They can not output multi-vector embeddings.
- When using retrieval and code models, you must add `Query: ` or `Passage: ` in front of the input. This ensure the query and retrieval targets are correctly embedded into the correct space.

## Multimodal Task-Specific Models

TBA

## Get Embeddings

First [install llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

Run `llama-server` to host the embedding model as OpenAI API compatible HTTP server:

```bash
llama-server -m jina-embeddings-v4-text-matching-F16.gguf --embedding --pooling mean
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
      "A beautiful sunset over the beach",
      "Un beau coucher de soleil sur la plage",
      "海滩上美丽的日落",
      "浜辺に沈む美しい夕日"
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
      "Query: 海滩上美丽的日落",
      "Query: 浜辺に沈む美しい夕日"
    ]
  }'
```

You can also use `llama-embedding` for one-shot embedding:

```bash
llama-embedding -m jina-embeddings-v4-text-matching-F16.gguf --pooling mean -p "jina is awesome"  2>/dev/null
```
