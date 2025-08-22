---
base_model:
- jinaai/jina-embeddings-v4
base_model_relation: quantized
---

# jina-embeddings-v4-gguf

A collection of GGUF and quantizations for [`jina-embeddings-v4`](https://huggingface.co/jinaai/jina-embeddings-v4).

> [!IMPORTANT]
> We highly recommend to first read [this blog post for more technical details and customized llama.cpp build](https://jina.ai/news/optimizing-ggufs-for-decoder-only-embedding-models).



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


Here's the speed and quality evaluation on two nano benchmarks. The higher the better. `IQ3_S` seems to be a good balance between size and speed.

#### Table 1: Tokens per Second on NanoHotpotQA `Documents`

| Quantization | BPW | File Size (GB) | Peak VRAM (GB) | Token/s w FA | Token/s w/o FA |
|------------------|-----------|-----|-----------|--------------|----------------|
| IQ1_S | 2.04 | 0.73 | 4.04 | 3625 | 2050 |
| IQ1_M | 2.19 | 0.79 | 4.09 | 3349 | 1997 |
| IQ2_XXS | 2.44 | 0.88 | 4.19 | 3701 | 2071 |
| IQ2_M | 2.94 | 1.06 | 4.37 | 3407 | 1989 |
| Q2_K | 3.29 | 1.18 | 4.49 | 3173 | 1905 |
| IQ3_XXS | 3.31 | 1.19 | 4.50 | 3668 | 2067 |
| IQ3_XS | 3.59 | 1.29 | 4.60 | 3604 | 2053 |
| IQ3_S | 3.76 | 1.35 | 4.66 | 3599 | 2049 |
| IQ3_M | 3.84 | 1.38 | 4.69 | 3603 | 2053 |
| Q3_K_M | 4.11 | 1.48 | 4.78 | 3450 | 2008 |
| IQ4_NL | 4.72 | 1.69 | 5.00 | 3571 | 2039 |
| IQ4_XS | 4.49 | 1.61 | 4.92 | 3585 | 2046 |
| Q4_K_M | 4.99 | 1.79 | 5.10 | 3558 | 2045 |
| Q5_K_S | 5.61 | 2.02 | 5.32 | 3567 | 2044 |
| Q5_K_M | 5.75 | 2.07 | 5.38 | 3528 | 2034 |
| Q6_K | 6.56 | 2.36 | 5.66 | 3334 | 1981 |
| Q8_0 | 8.50 | 3.05 | 6.36 | 3767 | 2101 |
| F16 | 16.00 | 5.75 | 9.70 | 3399 | 2023 |
| v3 (Transformers) | 16.00 | 1.10 | 2.82 | | 16505 |
| v4 (Transformers) | 16.00 | 7.40 | 14.45 | | 1865 |


System info:
```
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 36 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 37/37 layers to GPU
load_tensors:        CUDA0 model buffer size =  3127.61 MiB
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
...................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 4096
llama_context: n_ubatch      = 4096
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1  // 1 for w/ FA in the table; 0 for w/o FA
llama_context: kv_unified    = true
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (4096) < n_ctx_train (128000) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.59 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   144.00 MiB
llama_kv_cache_unified: size =  144.00 MiB (  4096 cells,  36 layers,  1/1 seqs), K (f16):   72.00 MiB, V (f16):   72.00 MiB
llama_context:      CUDA0 compute buffer size =  2470.16 MiB
llama_context:  CUDA_Host compute buffer size =    96.17 MiB
llama_context: graph nodes  = 1234
llama_context: graph splits = 2
common_init_from_params: added <|endoftext|> logit bias = -inf
common_init_from_params: added <|im_end|> logit bias = -inf
common_init_from_params: added <|fim_pad|> logit bias = -inf
common_init_from_params: added <|repo_name|> logit bias = -inf
common_init_from_params: added <|file_sep|> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 4096
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)

system_info: n_threads = 4 (n_threads_batch = 4) / 8 | CUDA : ARCHS = 890 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VNNI = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
main: n_tokens in batch = 0
main: number of embeddings = 5090
```

#### Table 2: NDCG@5
| Quantization | NanoHotpotQA | NanoFiQA2018 | NanoArguAna | NanoNFCorpus | NanoSciFact | Δ to v3 (HotpotQA) | Δ to v4 (HotpotQA) | Δ to v3 (FiQA2018) | Δ to v4 (FiQA2018) | Δ to v3 (ArguAna) | Δ to v4 (ArguAna) | Δ to v3 (NFCorpus) | Δ to v4 (NFCorpus) | Δ to v3 (SciFact) | Δ to v4 (SciFact) |
|------------------|--------------|--------------|-------------|--------------|-------------|-------------------|-------------------|-------------------|-------------------|------------------|------------------|-------------------|-------------------|------------------|------------------|
| IQ1_S | 0.6369 | 0.3178 | 0.3798 | 0.2933 | 0.5934 | -14% | -20% | -38% | -43% | -17% | -22% | -28% | -33% | -24% | -25% |
| IQ1_M | 0.6316 | 0.3313 | 0.5167 | 0.3256 | 0.6114 | -15% | -21% | -36% | -41% | +12% | +7% | -20% | -25% | -22% | -23% |
| IQ2_XXS | 0.7236 | 0.4582 | 0.4584 | 0.4067 | 0.7392 | -2% | -9% | -11% | -18% | -0% | -5% | -0% | -7% | -5% | -7% |
| IQ2_M | 0.7427 | 0.5869 | 0.5090 | 0.4468 | 0.7880 | +0% | -7% | +14% | +5% | +11% | +5% | +10% | +3% | +1% | -1% |
| Q2_K | 0.7683 | 0.5744 | 0.5168 | 0.4183 | 0.7546 | +4% | -4% | +12% | +3% | +12% | +7% | +3% | -4% | -4% | -5% |
| IQ3_XXS | 0.7780 | 0.5991 | 0.4811 | 0.4267 | 0.7610 | +5% | -2% | +16% | +8% | +5% | -1% | +5% | -2% | -3% | -4% |
| IQ3_XS | 0.7727 | 0.5615 | 0.5195 | 0.4439 | 0.7726 | +5% | -3% | +9% | +1% | +13% | +7% | +9% | +2% | -1% | -3% |
| IQ3_S | 0.8002 | 0.5505 | 0.4886 | 0.4381 | 0.7690 | +8% | +0% | +7% | -1% | +6% | +1% | +8% | +1% | -2% | -3% |
| IQ3_M | 0.8106 | 0.5387 | 0.5091 | 0.4462 | 0.7760 | +10% | +2% | +5% | -3% | +11% | +5% | +10% | +3% | -1% | -3% |
| Q3_K_M | 0.7567 | 0.5267 | 0.4486 | 0.4092 | 0.7775 | +2% | -5% | +2% | -5% | -2% | -7% | +1% | -6% | -1% | -2% |
| IQ4_NL | 0.7930 | 0.5598 | 0.4911 | 0.4285 | 0.7794 | +7% | -1% | +9% | +0% | +7% | +1% | +5% | -2% | -0% | -2% |
| IQ4_XS | 0.7979 | 0.5627 | 0.4947 | 0.4258 | 0.7789 | +8% | +0% | +9% | +1% | +8% | +2% | +5% | -2% | -0% | -2% |
| Q4_K_M | 0.8029 | 0.5569 | 0.4883 | 0.4226 | 0.7877 | +9% | +1% | +8% | +0% | +6% | +1% | +4% | -3% | +1% | -1% |
| Q5_K_S | 0.7969 | 0.5581 | 0.4721 | 0.4288 | 0.7842 | +8% | +0% | +8% | +0% | +3% | -3% | +5% | -1% | +0% | -2% |
| Q5_K_M | 0.7927 | 0.5601 | 0.4745 | 0.4247 | 0.7873 | +7% | -1% | +9% | +1% | +3% | -2% | +4% | -2% | +1% | -1% |
| Q6_K | 0.7951 | 0.5636 | 0.4822 | 0.4337 | 0.7846 | +8% | +0% | +10% | +1% | +5% | -0% | +7% | -0% | +0% | -1% |
| Q8_0 | 0.7938 | 0.5687 | 0.4784 | 0.4335 | 0.7851 | +7% | +0% | +11% | +2% | +4% | -1% | +7% | -0% | +0% | -1% |
| F16 | 0.7940 | 0.5610 | 0.4931 | 0.4343 | 0.7963 | +7% | +0% | +9% | +1% | +7% | +2% | +7% | -0% | +2% | +0% |
| v3 (Transformers) | 0.7393 | 0.5144 | 0.4600 | 0.4068 | 0.7820 | +0% | -7% | +0% | -8% | +0% | -5% | +0% | -6% | +0% | -2% |
| v4 (Transformers) | 0.7977 | 0.5571 | 0.4844 | 0.4351 | 0.7963 | +8% | +0% | +8% | +0% | +5% | +0% | +7% | +0% | +2% | +0% |


