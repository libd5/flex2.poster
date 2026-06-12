# 个性化海报/封面生成

## 环境配置
```
#先准备cuda支持的Pytorch环境 (以torch==2.9.1+cu128为例)
pip3 install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
#安装其它依赖
pip3 install -r requirements.txt
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
