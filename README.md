# AI Code Assistant Model

Một mô hình AI chuyên hỗ trợ lập trình (Code Generation Assistant) được fine-tune từ các mô hình ngôn ngữ lớn mã nguồn mở mạnh mẽ. Mô hình chuyên biệt cho việc **viết code**, **debug**, **giải thích code** và trả lời các câu hỏi liên quan đến lập trình, với khả năng hỗ trợ tốt tiếng Việt.

---

## 1. Giới thiệu

Dự án này hướng dẫn cách tạo ra một mô hình AI chuyên về code bằng cách **fine-tune** các mô hình nền mã nguồn mở (như Microsoft Phi-3-mini hoặc CodeLlama) trên dataset code/instruction.

### Mô hình sau khi fine-tune sẽ có khả năng:

- Viết code chính xác theo yêu cầu (Python, JavaScript, Java, C++, TypeScript, Go, Rust...)
- Giải thích code phức tạp một cách dễ hiểu
- Debug và sửa lỗi code
- Refactor code, tối ưu hiệu suất
- Hỗ trợ tốt tiếng Việt trong lập trình (giải thích, comment, tài liệu)
- Chạy được trên máy cá nhân có GPU hoặc Google Colab miễn phí

---

## 2. Mô hình nền khuyến nghị

| Mô hình | Kích thước | Ưu điểm | Khuyến nghị sử dụng |
|---------|------------|---------|---------------------|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B params | Nhỏ gọn, mạnh về code & reasoning, hỗ trợ tiếng Việt tốt, dễ fine-tune | Máy cá nhân / Colab miễn phí (từ 8GB VRAM) |
| `codellama/CodeLlama-7b-Instruct-hf` | 7B params | Chuyên sâu về code, chất lượng cao | GPU mạnh hơn (≥12–16GB VRAM) |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | 6.7B params | Rất mạnh về code generation, đa ngôn ngữ | GPU ≥12GB VRAM |
| `bigcode/starcoder2-7b` | 7B params | Đa ngôn ngữ lập trình tốt | GPU ≥12GB VRAM |

**Khuyến nghị chính:** Bắt đầu với **Phi-3-mini-4k-instruct** vì:
- Dễ chạy trên phần cứng phổ thông
- Chất lượng code generation tốt so với kích thước
- Cộng đồng hỗ trợ fine-tune rất nhiều (Unsloth, TRL, PEFT)

---

## 3. Tính năng nổi bật

- Sử dụng kỹ thuật **QLoRA** (4-bit quantization + LoRA) → fine-tune hiệu quả trên GPU thông thường
- Hỗ trợ dataset tùy chỉnh (code riêng của team / dự án)
- Inference nhanh, dễ deploy (GGUF, vLLM, Hugging Face, Ollama...)
- Prompt format theo kiểu instruction-tuning (dễ sử dụng như ChatGPT cho code)
- Có thể mở rộng sang hỗ trợ tiếng Việt mạnh hơn bằng cách trộn dataset tiếng Việt

---

## 4. Yêu cầu hệ thống

### Phần cứng tối thiểu

| Mô hình | VRAM khuyến nghị (QLoRA 4-bit) | Ghi chú |
|---------|-------------------------------|--------|
| Phi-3-mini (3.8B) | 8–10 GB | Chạy tốt trên RTX 3060/4060, Colab T4 |
| CodeLlama-7B / DeepSeek-Coder-6.7B | 12–16 GB | RTX 3080/4070 trở lên |

### Phần mềm

- Python 3.10+
- CUDA Toolkit tương thích với GPU
- Các thư viện chính:

```bash
pip install torch transformers datasets peft accelerate bitsandbytes trl
# Khuyến nghị thêm (tăng tốc đáng kể):
pip install unsloth  # hoặc dùng notebook Unsloth
# hoặc
pip install flash-attn --no-build-isolation  # nếu GPU hỗ trợ
```

---

## 5. Quy trình Fine-tune (QLoRA)

### 5.1. Cài đặt môi trường

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/Mac
# hoặc
venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate bitsandbytes trl
```

### 5.2. Load model với 4-bit quantization

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"  # nếu hỗ trợ
)
```

### 5.3. Cấu hình LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                        # Rank - tăng lên 32/64 nếu muốn chất lượng cao hơn
    lora_alpha=32,               # Thường = 2 * r
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "qkv_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"   # Phi-3
        # Với CodeLlama/Llama: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 5.4. Chuẩn bị Dataset

**Các dataset phổ biến:**

| Dataset | Mô tả | Link |
|---------|-------|------|
| `sahil2801/CodeAlpaca-20k` | Instruction → Code | Hugging Face |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | Code instruction chất lượng cao | Hugging Face |
| `bigcode/the-stack-smol` | Code thô đa ngôn ngữ | Hugging Face |
| Custom dataset | Code riêng của bạn | Tự tạo |

**Định dạng khuyến nghị (Alpaca / ChatML):**

```json
{
  "instruction": "Viết hàm Python tính giai thừa bằng đệ quy",
  "input": "",
  "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)"
}
```

Hoặc dạng messages (phù hợp với chat template của Phi-3):

```json
{
  "messages": [
    {"role": "user", "content": "Viết hàm Python tính giai thừa bằng đệ quy"},
    {"role": "assistant", "content": "def factorial(n):\n    ..."}
  ]
}
```

### 5.5. Training với TRL (SFTTrainer)

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./phi3-code-assistant",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    dataset_text_field="text",          # hoặc dùng formatting_func
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model("./phi3-code-assistant-lora")
```

---

## 6. Inference sau khi fine-tune

```python
from peft import PeftModel

# Load base model + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./phi3-code-assistant-lora")

# Generate
prompt = """<|user|>
Viết một class Python quản lý danh sách công việc (Todo List) với các phương thức thêm, xóa, đánh dấu hoàn thành.
<|end|>
<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.2,
    do_sample=True,
    top_p=0.9
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 7. Nâng cao hiệu suất & Tips

| Mẹo | Giải thích |
|-----|------------|
| Dùng **Unsloth** | Tăng tốc 2x, giảm VRAM ~50% |
| `r=16 → 32/64` | Tăng capacity của adapter nếu dữ liệu phức tạp |
| Trộn dataset tiếng Việt | Cải thiện khả năng giải thích code bằng tiếng Việt |
| Gradient checkpointing | Tiết kiệm VRAM |
| Packing = False | Tránh padding thừa làm chậm training |
| Đánh giá bằng HumanEval / MBPP | Đo chất lượng code generation thực tế |

### Dataset tiếng Việt gợi ý

- Vi-Alpaca
- OpenOrca-Viet
- Các dataset instruction tiếng Việt trên Hugging Face
- Tự tạo dataset từ code + giải thích tiếng Việt

---

## 8. Deploy mô hình

Sau khi fine-tune, bạn có thể:

1. **Merge LoRA** vào base model rồi export GGUF (dùng với Ollama / llama.cpp)
2. Deploy bằng **vLLM** hoặc **Text Generation Inference**
3. Upload adapter lên Hugging Face Hub
4. Tích hợp vào VS Code / Cursor qua API

Ví dụ merge:

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./phi3-code-assistant-merged")
tokenizer.save_pretrained("./phi3-code-assistant-merged")
```

---

## 9. Roadmap phát triển

- [ ] Fine-tune trên dataset code tiếng Việt chất lượng cao
- [ ] Hỗ trợ multi-turn conversation (debug qua nhiều lượt)
- [ ] Thêm khả năng code review & security analysis
- [ ] Quantize xuống GGUF Q4/Q5 để chạy trên CPU/Mac
- [ ] Tích hợp tool-calling (chạy code, search documentation...)

---

## 10. Tài liệu tham khảo

- [Microsoft Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Unsloth Phi-3 Fine-tune Notebook](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL SFTTrainer](https://huggingface.co/docs/trl)

---

**Tác giả dự án:** Cập nhật dựa trên nội dung gốc  
**Phiên bản:** 1.1  
**Cập nhật lần cuối:** 2026-09-02
```
