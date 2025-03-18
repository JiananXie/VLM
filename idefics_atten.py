import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
import numpy as np   
import math
import os
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image = load_image("view_336.jpg")

processor = AutoProcessor.from_pretrained("models/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "models/idefics2-8b",
     torch_dtype=torch.float16,    
).to(DEVICE)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What are the things I should be cautious about when I visit this place? Are there any dangerous areas or activities I should avoid? Or any other important information I should know?"},
        ]
    }      
]
# processor.image_processor.do_image_splitting = False
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
img_indices= (inputs['input_ids'] == model.config.image_token_id).nonzero()
img_start = img_indices[0][1].item()
img_end = img_indices[-1][1].item()
# print(inputs['input_ids'].size()) #[1, 371]
len_dict = {'sys': img_start - 1, 'img': img_end - img_start + 3, 'inst': inputs['input_ids'].size(1) - img_end - 1}
print(len_dict)
print(processor.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True))
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# Generate
output = model.generate(**inputs, use_cache=True, max_new_tokens=512, output_attentions=True, return_dict_in_generate=True)
output_ids = output.sequences
attentions = output.attentions #(generate_length, num_layers, [batch_size, num_heads, seq_length, seq_length])

print(output_ids.size())
generated_texts = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(len(attentions))
print(len(attentions[0]))
print(attentions[0][0].size())
print(attentions[-1][0].size())
seq_len = output_ids.size(1)
len_dict['out'] = seq_len -  inputs['input_ids'].size(1)

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
save_dir = f'result/idefics2'
os.makedirs(save_dir, exist_ok=True)
for idx, layer in tqdm(enumerate(attentions_maps), desc="Drawing heat maps"):
    # plt.subplot(1, 3, idx + 1)
    fig = plt.figure(figsize=(20, 10)) 
    attn_map = attentions_maps[layer]
    # attn_map = torch.nn.functional.avg_pool2d(attn_map.unsqueeze(0).unsqueeze(0), stride=interval).squeeze(0).squeeze(0).cpu().numpy()
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
        # xticks.append(math.ceil((tick_idx + v) / interval))
        xticks.append(tick_idx + v)
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



print(generated_texts)