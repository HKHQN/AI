# AI
Document AI 

# AI Code Assistant Model

Một mô hình AI chuyên hỗ trợ lập trình (code generation assistant) được fine-tune từ mô hình nền mạnh mẽ, dành riêng cho việc viết code, debug, giải thích code và trả lời các câu hỏi liên quan đến lập trình.

## Giới thiệu

Dự án này hướng dẫn cách tạo ra một mô hình AI chuyên về code bằng cách **fine-tune** các mô hình ngôn ngữ lớn mã nguồn mở (như Microsoft Phi-3-mini hoặc CodeLlama) trên dataset code/instruction. 

Mô hình sau khi fine-tune sẽ:
- Viết code chính xác theo yêu cầu (Python, JavaScript, Java, C++, v.v.)
- Giải thích code phức tạp
- Debug và sửa lỗi code
- Hỗ trợ tốt tiếng Việt trong lập trình
- Chạy được trên máy cá nhân có GPU hoặc Google Colab

## Mô hình nền khuyến nghị

| Mô hình                        | Kích thước     | Ưu điểm                                      | Khuyến nghị sử dụng                  |
|-------------------------------|----------------|----------------------------------------------|--------------------------------------|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B params   | Nhỏ gọn, mạnh về code & reasoning, hỗ trợ tiếng Việt tốt | Máy cá nhân / Colab miễn phí        |
| `codellama/CodeLlama-7b-Instruct-hf` | 7B params    | Chuyên sâu về code, chất lượng cao           | GPU mạnh hơn (≥16GB VRAM)            |

## Tính năng nổi bật

- Sử dụng kỹ thuật **QLoRA** (4-bit quantization + LoRA) để fine-tune hiệu quả trên GPU thông thường
- Hỗ trợ dataset tùy chỉnh (có thể dùng code riêng của bạn)
- Inference nhanh, dễ deploy
- Prompt format theo kiểu instruction-tuning (dễ sử dụng như ChatGPT cho code)

## Yêu cầu hệ thống

- Python 3.10+
- GPU NVIDIA (tối thiểu 8GB VRAM cho Phi-3-mini)
- Các thư viện chính:
  ```bash
  pip install torch transformers datasets peft accelerate bitsandbytes trl
