# 如何准备训练数据
我们在`huggingface`仓库`libadi/flex2.poster`的`/data`目录下提供了原生训练数据，或者按照以下步骤准备数据
## 根据关键词爬取互联网图片
我们使用关键词`poster book course cover`从`Bing`爬取一定数量的图片，请运行
```bash
 python data_toolkits/crawl.py   --keyword "poster book course cover"   --data_root ./data/images   --num_samples 600   --delay 1.0
```
## 根据分辨率筛选数据
我们过滤掉其中一部分分辨率非常低的数据
```bash
python data_toolkits/filter_by_res.py --data_root ./data/images --min_resolution 512
```

## 提示词生成
我们本地部署`Qwen3.5-27B`模型为我们生成图片的描述作为提示词
```bash
python data_toolkits/caption.py --data_path ./data/images --output_path ./data/captions --model_name Qwen/Qwen3.5-27B
python data_toolkits/caption.py --data_path ./data/images --output_path ./data/captions_zh --language zh --model_name Qwen/Qwen3.5-27B
```

## 双语提示词 JSON
将英文 `captions/` 与中文 `captions_zh/` 合并为JSON（`caption`=英文，`caption_short`=中文）：
```bash
python data_toolkits/build_bilingual_captions.py --data_root ./data --output_path ./data/captions_bilingual.json
```
