"""
Caption the crawled images with locally deployed Qwen3.5 
Alternatives can be API call. 
"""
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import argparse 
from tqdm import tqdm 
import os 
import glob 
import re
SYSTEM_PROMPT = "You are an expert in poster design. You would be given an image, likely illustrating a poster or book/course cover. Your task is to describe the layout and content of the image in detail by natural language, which would be used as the text prompt for generating such poster. Please answer within 100 words."
SYSTEM_PROMPT_ZH = "你是一个海报设计专家。你会得到一张图片，可能是一张海报，书籍封面或课程封面等。你的任务是用自然语言详细描述图片的布局和内容，之后将作为生成海报的文本提示。请在100个词以内回答。"
THINKING_CLOSE_TAG = "</" + "think" + ">"

def strip_thinking(text: str) -> str:
    """Keep only the final answer, dropping Qwen thinking/reasoning blocks."""
    if THINKING_CLOSE_TAG in text:
        text = text.split(THINKING_CLOSE_TAG, 1)[1]
    text = re.sub(
        r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>",
        "",
        text,
        count=1,
    )
    return text.strip()

def worker(args, tasks, device='cuda'):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name)   

    for task in tqdm(tasks, desc='Captioning images'):
        messages = [
            {
                "role": "system", "content": SYSTEM_PROMPT_ZH if args.language == 'zh' else SYSTEM_PROMPT
            },
            {
                "role": "user", "content": [
                    {
                        "type": "image", "image": task['image']
                    },
                    {"type": "text", "text": "描述图片" if args.language == 'zh' else "Describe the image."}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output_text = strip_thinking(output_text)
        os.makedirs(os.path.dirname(task['output_path']), exist_ok=True)
        with open(task['output_path'], 'w') as f:
            f.write(output_text)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])
    args = parser.parse_args()

    images = glob.glob(os.path.join(args.data_path, '*.jpg'))
    tasks = [{'image': image, 'output_path': os.path.join(args.output_path, os.path.basename(image).split('.')[0] + '.txt')} for image in images]

    worker(args, tasks)