import argparse
import torch
import math
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns   
from tqdm import tqdm
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, attn_implementation="eager"
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images] # [(1000, 667)]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print(images_tensor.shape)
    print(image_sizes)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    # print(input_ids.shape)
    # print(input_ids)
    img_idx = torch.where(input_ids.squeeze(0) == -200)[0].item()
    len_dict = {}
    len_dict['sys'] = img_idx

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True
        )
    output_ids = output.sequences
    attentions = output.attentions #(generate_length, num_layers, [batch_size, num_heads, seq_length, seq_length])
    
    seq_len = attentions[0][0].shape[3] + len(attentions)
    img_len = seq_len - len_dict['sys'] - (input_ids.shape[1] - img_idx - 1) - len(attentions)
    len_dict['img'] = img_len
    len_dict['inst'] = input_ids.shape[1] - img_idx -1 
    len_dict['out'] = len(attentions)
    print(f"total seq length: {seq_len}")
    print(f"input length: {attentions[0][0].shape[3]}")
    print(len_dict)
    print(attentions[-1][0].shape)
    #attention map
    attentions_maps = {}
    layer_range = range(0, 32, 1)
    for layer in layer_range:
        attentions_maps[layer + 1] = torch.zeros((seq_len-1, seq_len-1))

        step = 0
        for i in tqdm(range(len(attentions)), desc=f"Processing layer {layer + 1}"):
            attention = attentions[i][layer].squeeze(0)  # 去掉batch维度=1, [num_heads, seq_length, seq_length]
            attention = attention.mean(dim=0)  # 多头取均值, [seq_length, seq_length]
            for j in range(attention.shape[0]):
                for k in range(attention.shape[1]):
                    attentions_maps[layer + 1][step + j][k] = attention[j][k]
            step += attention.shape[0]

        # pd.DataFrame(attentions_maps[layer + 1]).to_csv(f'attentions_{layer + 1}.csv', index=False, header=False)
    global_min = 1
    global_max = 0
    for layer in attentions_maps.keys():
        temp = torch.nn.functional.avg_pool2d(attentions_maps[layer].unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0).cpu().numpy()
        global_min = min(global_min, np.min(temp[temp > 0]))  # 找到所有里面最小的
        global_max = max(global_max, np.max(temp))  # 找到所有里面最大的
    print(global_min, global_max)
    if not os.path.exists('result'):
        os.makedirs('result')
    save_dir = f'result/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    for idx, layer in tqdm(enumerate(attentions_maps), desc="Drawing heat maps"):
        # plt.subplot(1, 3, idx + 1)
        fig = plt.figure(figsize=(20, 10)) 
        aspect_ratio = model.config.image_aspect_ratio
        if aspect_ratio == 'anyres':
            interval = 40
        else:   
            interval = 20
        attn_map = attentions_maps[layer]
        attn_map = torch.nn.functional.avg_pool2d(attn_map.unsqueeze(0).unsqueeze(0), interval, stride=interval).squeeze(0).squeeze(0).cpu().numpy()
        log_norm = LogNorm(vmin=global_min, vmax=1)
        mask = np.triu(np.ones_like(attn_map), k=1)
        heatmap = sns.heatmap(attn_map, cmap='viridis', 
                              cbar=True, mask=mask, square=True, 
                              norm=log_norm,
                              cbar_kws={'label': 'Attention Score'})  # 保存热图对象
        
        plt.title(f'Layer {layer}')
        xticks = []
        tick_idx = 0
        for _, v in list(len_dict.items())[:-1]:
            xticks.append(math.ceil((tick_idx + v) / interval))
            tick_idx += v
        yticks = list(reversed(xticks))
        plt.xticks(xticks)
        plt.yticks(yticks)
        if not os.path.exists(f'{save_dir}/layer_{layer}'):
            os.makedirs(f'{save_dir}/layer_{layer}')
        plt.savefig(f'{save_dir}/layer_{layer}/attn_map_{layer}.png', dpi=400, bbox_inches='tight') 
        plt.close()

    for idx, layer in tqdm(enumerate(attentions_maps), desc="Drawing pie charts"):
        fig = plt.figure(figsize=(20, 10)) 
        attn_map = attentions_maps[layer].cpu().numpy()[-len_dict['out']:, :]
        attn_map = np.sum(attn_map, axis=0)
        # attn_map = attn_map / np.sum(attn_map)
        eff_dict = {}
        idx = 0
        for t in list(len_dict.keys()):
            eff_dict[t] = np.sum(attn_map[idx:idx+len_dict[t]]) / len_dict[t]
            idx += len_dict[t]
        plt.pie(eff_dict.values(), labels=eff_dict.keys(), autopct='%1.1f%%', colors=sns.color_palette("hls", len(eff_dict)))
        plt.legend(title="Token type", loc="upper right")
        plt.title(f'Attention efficiency distribution of layer {layer}')
        if not os.path.exists(f'{save_dir}/layer_{layer}'):
            os.makedirs(f'{save_dir}/layer_{layer}')
        plt.savefig(f'{save_dir}/layer_{layer}/attn_pie_{layer}.png', dpi=400, bbox_inches='tight') 
        plt.close()

    for idx, layer in tqdm(enumerate(attentions_maps), desc="Drawing visual attention distribution"):
        fig = plt.figure(figsize=(20, 10)) 
        attn_map = attentions_maps[layer].cpu().numpy()[-len_dict['out']-len_dict['inst']-len_dict['img']:,len_dict['sys']:len_dict['sys']+len_dict['img']]
        print(attn_map.shape)
        visual_atten_bars = {}
        idx = 0
        for t in list(len_dict.keys())[1:]:
            if t == 'img':
                visual_atten_bars[t] = np.sum(attn_map[idx:idx+len_dict[t],:], axis=0) / np.array(range(len_dict[t], 0, -1))
            else:
                visual_atten_bars[t] = np.mean(attn_map[idx:idx+len_dict[t],:], axis=0)
            idx += len_dict[t]
            plt.bar(range(len_dict['img']), visual_atten_bars[t])
            plt.xticks()
            # plt.grid(True)
            plt.title(f'[{t}] Attention Distribution')
            if not os.path.exists(f'{save_dir}/layer_{layer}'):
                os.makedirs(f'{save_dir}/layer_{layer}')
            plt.savefig(f'{save_dir}/layer_{layer}/attn_bar_{layer}_{t}.png', dpi=400, bbox_inches='tight') 
            plt.close()



    outputs = tokenizer.batch_decode(output_ids)[0].strip()
    print(f"Generation: {outputs}")




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
