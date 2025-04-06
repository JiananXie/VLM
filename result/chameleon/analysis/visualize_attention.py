import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm

# Load the JSON data
with open('/Users/wells/Desktop/Research/VLM/result/chameleon/combined_attention_metrics.json', 'r') as file:
    data = json.load(file)

# Extract the number of layers
num_layers = len(data)
layers = [int(layer.split('_')[1]) for layer in data.keys()]
layers.sort()

# Define attention types to analyze
attention_types = [
    'text_to_text', 'text_to_image', 
    'image_to_image', 
    'output_to_text', 'output_to_image'
]

# Create a DataFrame for easier manipulation
df = pd.DataFrame()

for layer in layers:
    layer_key = f'layer_{layer}'
    for att_type in attention_types:
        # Skip if attention type doesn't exist for this layer
        if att_type not in data[layer_key]['metrics']:
            continue
        
        mean = data[layer_key]['metrics'][att_type]['mean']
        max_val = data[layer_key]['metrics'][att_type]['max']
        min_val = data[layer_key]['metrics'][att_type]['min']
        std = data[layer_key]['metrics'][att_type]['std']
        
        df = pd.concat([df, pd.DataFrame({
            'Layer': layer,
            'Attention Type': att_type,
            'Mean': mean,
            'Max': max_val,
            'Min': min_val,
            'Std': std
        }, index=[0])], ignore_index=True)

# Set the style
plt.style.use('ggplot')

# Create a figure with multiple plots
plt.figure(figsize=(18, 10))

# 1. Heatmap of mean attention values across layers and types
plt.subplot(2, 2, 1)
heatmap_data = df.pivot(index='Attention Type', columns='Layer', values='Mean')
sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt='.4f', cbar_kws={'label': 'Mean Attention'})
plt.title('Mean Attention Values Across Layers', fontsize=14)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Attention Type', fontsize=12)

# 2. Line plot of mean attention for each type across layers
plt.subplot(2, 2, 2)
for att_type in attention_types:
    if att_type in df['Attention Type'].unique():
        subset = df[df['Attention Type'] == att_type]
        plt.plot(subset['Layer'], subset['Mean'], marker='o', label=att_type)
plt.legend(loc='best')
plt.title('Mean Attention Values by Type Across Layers', fontsize=14)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Mean Attention Value', fontsize=12)
plt.grid(True, alpha=0.3)

# 3. Heatmap of max attention values (using log scale for better visualization)
plt.subplot(2, 2, 3)
heatmap_data = df.pivot(index='Attention Type', columns='Layer', values='Max')
sns.heatmap(heatmap_data, cmap='plasma', annot=False, norm=LogNorm(), cbar_kws={'label': 'Max Attention (log scale)'})
plt.title('Max Attention Values Across Layers (Log Scale)', fontsize=14)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Attention Type', fontsize=12)

# 4. Line plot of std deviation for each attention type
plt.subplot(2, 2, 4)
for att_type in attention_types:
    if att_type in df['Attention Type'].unique():
        subset = df[df['Attention Type'] == att_type]
        plt.plot(subset['Layer'], subset['Std'], marker='o', label=att_type)
plt.legend(loc='best')
plt.title('Standard Deviation of Attention Values', fontsize=14)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Standard Deviation', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attention_metrics_visualization.png', dpi=300, bbox_inches='tight')

# Create a second figure for boxplots of different attention types
plt.figure(figsize=(18, 10))

# Melt the dataframe for violin plots
melted_df = pd.melt(df, id_vars=['Layer', 'Attention Type'], 
                    value_vars=['Mean', 'Max', 'Min', 'Std'],
                    var_name='Metric', value_name='Value')

# Create violin plots for each attention type
for i, att_type in enumerate(attention_types, 1):
    if att_type in df['Attention Type'].unique():
        plt.subplot(2, 3, i)
        att_data = melted_df[(melted_df['Attention Type'] == att_type) & (melted_df['Metric'] == 'Mean')]
        
        # Plot violin plot
        sns.violinplot(x='Layer', y='Value', data=att_data, inner='box', palette='Set3')
        plt.title(f'{att_type} Mean Attention Distribution', fontsize=12)
        plt.xlabel('Layer', fontsize=10)
        plt.ylabel('Mean Attention Value', fontsize=10)
        plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('attention_type_distributions.png', dpi=300, bbox_inches='tight')

# Create radar charts to compare attention patterns across layers
plt.figure(figsize=(15, 15))

# Select a few representative layers to compare
representative_layers = [0, 7, 15, 23, 31]
num_att_types = len(attention_types)

for i, layer in enumerate(representative_layers, 1):
    plt.subplot(2, 3, i, polar=True)
    
    # Get data for this layer
    layer_data = df[df['Layer'] == layer]
    
    # Extract values for each attention type
    values = []
    for att_type in attention_types:
        if att_type in layer_data['Attention Type'].values:
            values.append(layer_data[layer_data['Attention Type'] == att_type]['Mean'].values[0])
        else:
            values.append(0)
    
    # Close the radar chart by repeating first value
    values.append(values[0])
    
    # Compute angle for each attention type
    angles = np.linspace(0, 2*np.pi, len(attention_types), endpoint=False).tolist()
    angles.append(angles[0])  # Close the loop
    
    # Plot the radar chart
    plt.polar(angles, values, marker='o', label=f'Layer {layer}')
    plt.fill(angles, values, alpha=0.25)
    
    # Set the labels
    plt.xticks(angles[:-1], attention_types, fontsize=8)
    plt.title(f'Layer {layer} Attention Pattern', fontsize=12)

plt.tight_layout()
plt.savefig('attention_radar_charts.png', dpi=300, bbox_inches='tight')

print("Visualizations created successfully!") 