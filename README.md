# 个性化海报/封面生成

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
