#!/bin/bash

# Exit immediately if any command fails
set -e

# F16 model file path (passed as parameter)
F16_MODEL_FILE="$1"
OUTPUT_DIR="$2"
IMATRIX="$3"
HF_REPO="$4"

# Get base name by removing both -F16.gguf and .gguf patterns
FILENAME="$(basename "$F16_MODEL_FILE")"
BASE_NAME="${FILENAME%-F16.gguf}"
BASE_NAME="${BASE_NAME%.gguf}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array of quantization types
QUANT_TYPES=("IQ1_S" "IQ1_M" "IQ2_XXS" "IQ2_M" "Q2_K" "IQ4_NL" "IQ4_XS" "Q3_K_M" "Q4_K_M" "Q5_K_S" "Q5_K_M" "Q6_K" "Q8_0")

# Run quantizations sequentially
echo "Starting quantization..."
for quant_type in "${QUANT_TYPES[@]}"; do
    echo "Starting quantization: $quant_type"
    llama-quantize --imatrix "${IMATRIX}" "$F16_MODEL_FILE" "${OUTPUT_DIR}/${BASE_NAME}-${quant_type}.gguf" $quant_type 8
    echo "Uploading ${BASE_NAME}-${quant_type}.gguf to $HF_REPO"
    huggingface-cli upload "$HF_REPO" "${OUTPUT_DIR}/${BASE_NAME}-${quant_type}.gguf"
    echo "Deleting ${BASE_NAME}-${quant_type}.gguf to save space"
    rm "${OUTPUT_DIR}/${BASE_NAME}-${quant_type}.gguf"
done