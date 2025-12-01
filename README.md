# temp1

This repository contains a text reranker implementation using a CrossEncoder model.

## Files

- `reranker1.py` - Python implementation using sentence-transformers library
- `reranker1.cpp` - C++ implementation using ONNX Runtime

## Python Version

### Prerequisites

```bash
pip install sentence-transformers
```

### Usage

```bash
python reranker1.py
```

## C++ Version

The C++ version uses ONNX Runtime to load and run the CrossEncoder model.

### Prerequisites

1. **ONNX Runtime**: Download from [GitHub Releases](https://github.com/microsoft/onnxruntime/releases)
2. **ONNX Model**: Export the `cross-encoder/ms-marco-MiniLM-L6-v2` model to ONNX format
3. **Vocabulary File**: Get `vocab.txt` from the model files

### Building

```bash
# Configure with ONNX Runtime path
cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime

# Build
cmake --build build
```

### Usage

```bash
./build/reranker1 <model.onnx> <vocab.txt>
```

### Exporting the Model to ONNX

You can export the CrossEncoder model to ONNX format using Python:

```python
from sentence_transformers import CrossEncoder
import torch

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# Export to ONNX
dummy_input = {
    "input_ids": torch.zeros(1, 128, dtype=torch.long),
    "attention_mask": torch.ones(1, 128, dtype=torch.long),
    "token_type_ids": torch.zeros(1, 128, dtype=torch.long),
}

torch.onnx.export(
    model.model,
    (dummy_input,),
    "model.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)
```

The vocabulary file (`vocab.txt`) can be found in the model's directory after downloading:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
tokenizer.save_vocabulary(".")
```
