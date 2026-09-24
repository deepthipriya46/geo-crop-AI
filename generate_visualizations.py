import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure results folder exists
os.makedirs('results', exist_ok=True)

# Set styling
sns.set_theme(style="whitegrid", rc={"axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})
plt.rcParams['font.size'] = 14

def plot_crop_distribution(df):
    plt.figure(figsize=(10, 6))
    order = df['crop'].value_counts().index
    sns.countplot(y='crop', data=df, order=order, palette='viridis')
    plt.title('2. Crop Distribution (Number of Samples per Crop)')
    plt.xlabel('Number of Samples')
    plt.ylabel('Crop')
    plt.tight_layout()
    plt.savefig('results/2_crop_distribution.png', dpi=300)
    plt.close()

def plot_soil_crop_heatmap():
    sys.path.append(os.path.join(os.getcwd(), 'data'))
    try:
        from crop_database import CROP_DATABASE
    except ImportError:
        print("Could not import CROP_DATABASE")
        return
    
    crops = list(CROP_DATABASE.keys())
    soils = set()
    for c in crops:
        soils.update(CROP_DATABASE[c]['soil'])
    soils = list(soils)
    
    matrix = np.zeros((len(soils), len(crops)))
    for j, c in enumerate(crops):
        for i, s in enumerate(soils):
            if s in CROP_DATABASE[c]['soil']:
                matrix[i, j] = 1
                
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, xticklabels=crops, yticklabels=soils, cmap='Blues', cbar=False, linewidths=.5, linecolor='gray')
    plt.title('3. Soil-to-Crop Mapping Heatmap')
    plt.xlabel('Crop')
    plt.ylabel('Soil Type')
    plt.tight_layout()
    plt.savefig('results/3_soil_to_crop_heatmap.png', dpi=300)
    plt.close()
    return CROP_DATABASE

def plot_correlation_heatmap(df):
    plt.figure(figsize=(8, 6))
    corr = df[['temperature', 'humidity', 'rainfall', 'suitable']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('4. Weather Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/4_correlation_heatmap.png', dpi=300)
    plt.close()

def plot_suitability_dist(df):
    plt.figure(figsize=(6, 6))
    counts = df['suitable'].value_counts()
    labels = ['Unsuitable (0)' if idx == 0 else 'Suitable (1)' for idx in counts.index]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
    plt.title('12. Crop Suitability Distribution')
    plt.tight_layout()
    plt.savefig('results/12_crop_suitability_distribution.png', dpi=300)
    plt.close()

def plot_scatter_temp_rainfall(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='temperature', y='rainfall', hue='suitable', data=df, palette='Set1', alpha=0.7)
    plt.title('20. Scatter Plot: Temperature vs Rainfall')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Rainfall (mm)')
    plt.tight_layout()
    plt.savefig('results/20_scatter_temp_rainfall.png', dpi=300)
    plt.close()

def plot_scatter_hum_temp(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='humidity', y='temperature', hue='suitable', data=df, palette='Set1', alpha=0.7)
    plt.title('21. Scatter Plot: Humidity vs Temperature')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.savefig('results/21_scatter_humidity_temp.png', dpi=300)
    plt.close()

def parse_validation_logs():
    valid_soil = 0
    rejected_non_soil = 0
    blurry = 0
    
    data = []
    
    try:
        with open('logs/validation_log.csv', 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 3: continue
                status = parts[2]
                reason = parts[-1].lower()
                
                if 'valid_soil' in status.lower():
                    valid_soil += 1
                    # Extract soil type and confidence
                    for i in range(len(parts)):
                        if 'Soil' in parts[i]:
                            try:
                                conf = float(parts[i-1])
                                data.append({'Soil Type': parts[i], 'Confidence (%)': conf})
                            except: pass
                            break
                elif 'rejected' in status.lower():
                    if 'blur' in reason:
                        blurry += 1
                    else:
                        rejected_non_soil += 1
    except Exception as e:
        print("Error parsing validation log:", e)
        
    return valid_soil, rejected_non_soil, blurry, pd.DataFrame(data)

def plot_validation_summary(valid, rejected, blurry):
    plt.figure(figsize=(7, 7))
    counts = [valid, rejected, blurry]
    labels = ['Valid Soil Images', 'Rejected Non-Soil Images', 'Blurry Images']
    # filter out zeros
    c_filt, l_filt = [], []
    for c, l in zip(counts, labels):
        if c > 0:
            c_filt.append(c)
            l_filt.append(l)
            
    plt.pie(c_filt, labels=l_filt, autopct='%1.1f%%', colors=['#4CAF50','#F44336','#FFC107'], startangle=140)
    plt.title('23. Validation Result Summary')
    plt.tight_layout()
    plt.savefig('results/23_validation_result_summary.png', dpi=300)
    plt.close()

def plot_soil_confidence(df):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Soil Type', y='Confidence (%)', data=df, palette='Set2')
    plt.title('Soil Classification Confidence per Class')
    plt.xlabel('Soil Type')
    plt.ylabel('Confidence (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/XX_soil_classification_confidence.png', dpi=300)
    plt.close()

def plot_ideal_vs_real_temp(df, crop_db):
    if not crop_db: return
    df = df.copy()
    def get_ideal_avg(crop):
        if crop in crop_db:
            return (crop_db[crop]['ideal_temp_min'] + crop_db[crop]['ideal_temp_max']) / 2.0
        return np.nan
        
    df['ideal_temp'] = df['crop'].apply(get_ideal_avg)
    df = df.dropna(subset=['ideal_temp'])
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='ideal_temp', y='temperature', hue='suitable', data=df, palette='coolwarm', alpha=0.7)
    
    # Reference line y = x
    min_val = min(df['ideal_temp'].min(), df['temperature'].min())
    max_val = max(df['ideal_temp'].max(), df['temperature'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y=x (Ideal=Real)')
    
    plt.title('1. Scatter Plot: Ideal Temperature vs Real-Time Temperature')
    plt.xlabel('Ideal Temperature (°C)')
    plt.ylabel('Real-Time Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/1_ideal_vs_real_temp.png', dpi=300)
    plt.close()

def plot_box_temp_per_crop(df, crop_db):
    if not crop_db: return
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='crop', y='temperature', data=df, palette='Set3')
    
    # Add ideal temp markers
    crops = df['crop'].unique()
    for i, crop in enumerate(sorted(crops)): # sns boxplot sorts x alphabetically usually, let's just use order
        pass # Wait, let's specify order explicitly to match markers
        
    order = sorted(df['crop'].unique())
    plt.clf()
    sns.boxplot(x='crop', y='temperature', data=df, order=order, palette='Set3')
    
    for i, crop in enumerate(order):
        if crop in crop_db:
            ideal_min = crop_db[crop]['ideal_temp_min']
            ideal_max = crop_db[crop]['ideal_temp_max']
            plt.plot([i-0.4, i+0.4], [ideal_min, ideal_min], color='red', linestyle='-', linewidth=2)
            plt.plot([i-0.4, i+0.4], [ideal_max, ideal_max], color='red', linestyle='-', linewidth=2)
            if i == 0:
                plt.plot([],[], color='red', linestyle='-', linewidth=2, label='Ideal Temp Range')
                
    plt.title('4. Box Plot: Real Temperatures for Every Crop')
    plt.xlabel('Crop')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/4_box_temp_per_crop.png', dpi=300)
    plt.close()

def plot_temp_difference_heatmap(df, crop_db):
    if not crop_db: return
    
    crops = sorted(df['crop'].unique())
    data = []
    for c in crops:
        if c in crop_db:
            real_avg = df[df['crop'] == c]['temperature'].mean()
            ideal_avg = (crop_db[c]['ideal_temp_min'] + crop_db[c]['ideal_temp_max']) / 2.0
            diff = real_avg - ideal_avg
            data.append({
                'Crop': c.capitalize(),
                'Ideal Temperature': ideal_avg,
                'Real Temperature': real_avg,
                'Temperature Difference': diff
            })
            
    if not data: return
    
    heat_df = pd.DataFrame(data).set_index('Crop')
    
    plt.figure(figsize=(8, 10))
    sns.heatmap(heat_df, annot=True, cmap='RdBu_r', fmt=".2f", center=0, linewidths=.5, linecolor='gray')
    plt.title('Temperature Difference Heatmap')
    plt.tight_layout()
    plt.savefig('results/temperature_difference_heatmap.png', dpi=300)
    plt.close()

def plot_performance_metrics_heatmap():
    metrics = []
    classes = []
    try:
        with open('results/evaluation_report.txt', 'r') as f:
            lines = f.readlines()
            
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('precision') and 'recall' in line:
                start_idx = i + 2
                break
                
        if start_idx != -1:
            for line in lines[start_idx:]:
                if not line.strip(): continue
                parts = line.split()
                if len(parts) >= 4 and '_Soil' in parts[0]:
                    classes.append(parts[0].replace('_', ' '))
                    metrics.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
        if metrics:
            heat_df = pd.DataFrame(metrics, index=classes, columns=['Precision', 'Recall', 'F1-Score'])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(heat_df, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, linecolor='gray')
            plt.title('Performance Metrics Heatmap for Soil Classification')
            plt.xlabel('Evaluation Metrics')
            plt.ylabel('Soil Classes')
            plt.tight_layout()
            plt.savefig('results/performance_metrics_heatmap.png', dpi=300)
            plt.close()
    except Exception as e:
        print("Error parsing evaluation report:", e)

if __name__ == '__main__':
    df = pd.read_csv('data/crop_weather_dataset.csv')
    
    print("Plotting Crop Distribution...")
    plot_crop_distribution(df)
    
    print("Plotting Soil-to-Crop Heatmap...")
    crop_db = plot_soil_crop_heatmap()
    
    print("Plotting Correlation Heatmap...")
    plot_correlation_heatmap(df)
    
    print("Plotting Suitability Distribution...")
    plot_suitability_dist(df)
    
    print("Plotting Scatter Temp vs Rainfall...")
    plot_scatter_temp_rainfall(df)
    
    print("Plotting Scatter Humidity vs Temp...")
    plot_scatter_hum_temp(df)
    
    print("Parsing Validation Logs...")
    valid, rejected, blurry, conf_df = parse_validation_logs()
    
    print("Plotting Validation Summary...")
    plot_validation_summary(valid, rejected, blurry)
    
    print("Plotting Soil Confidence...")
    plot_soil_confidence(conf_df)
    
    print("Plotting Ideal vs Real Temp...")
    plot_ideal_vs_real_temp(df, crop_db)
    
    print("Plotting Box Temp per Crop...")
    plot_box_temp_per_crop(df, crop_db)
    
    print("Plotting Temperature Difference Heatmap...")
    plot_temp_difference_heatmap(df, crop_db)
    
    print("Plotting Performance Metrics Heatmap...")
    plot_performance_metrics_heatmap()
    
    print("All visualizations generated successfully in the 'results' folder.")
