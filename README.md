# 个性化海报/封面生成

## 环境配置
```
#先准备cuda支持的Pytorch环境 (以torch==2.9.1+cu128为例)
pip3 install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
#安装其它依赖
pip3 install -r requirements.txt
```

如果使用 `uv`：

```bash
uv venv
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
uv pip install -r requirements.txt
```
## 训练数据准备

见 `data_toolkits/`

## 训练

```bash
cd src
python run.py config/flex2_lora_poster.yaml
```

## 推理

```bash
cd src

# 基座模型
python infer.py --prompt "minimalist book sale poster with bold typography" \
  --canny ../data/canny/<hash>.jpg --output_dir ./outputs/base

python infer.py --prompt "极简风格书籍促销海报" \
  --canny ../data/canny/<hash>.jpg \
  --lora libadi/flex2.poster --output_dir ./outputs/lora

```

## Web UI

本项目已提供 Gradio 前端，默认加载训练好的 LoRA，并在页面中暴露 prompt、Canny 控制图、长宽、seed、guidance、steps 和 LoRA scale。

```bash
cd src
uv run python app.py
```

浏览器打开 `http://127.0.0.1:7860`。

常用参数：

```bash
# 只用基座模型
uv run python app.py --no-lora

# 指定 LoRA 文件或 Hugging Face 仓库
uv run python app.py --lora ../ckpts/lora_model.safetensors
uv run python app.py --lora libadi/flex2.poster

# 改端口或允许局域网访问
uv run python app.py --host 0.0.0.0 --port 7861
```
