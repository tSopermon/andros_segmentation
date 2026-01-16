import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_history(path='outputs/training_history.npy'):
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.size == 1:
            return data.item()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def plot_comparison(history, output_dir='outputs'):
    metrics = [
        ('val_loss', 'Validation Loss', 'lower'),
        ('val_iou_mean', 'Validation mIoU', 'higher'),
        ('val_f1_mean', 'Validation F1 Score', 'higher')
    ]
    
    plt.figure(figsize=(15, 12))
    sns.set_style("whitegrid")
    
    for i, (metric_key, title, better) in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        
        for model_name, metrics_dict in history.items():
            if metric_key in metrics_dict:
                values = metrics_dict[metric_key]
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=f"{model_name}", linewidth=2, alpha=0.8)
                
        plt.title(f'Model Comparison - {title}', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_comparison_metrics.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved comparison plot to {save_path}")
    plt.close()

def plot_individual_models(history, output_dir='outputs'):
    for model_name, metrics_dict in history.items():
        plt.figure(figsize=(15, 5))
        sns.set_style("whitegrid")
        
        # Plot Loss
        plt.subplot(1, 3, 1)
        if 'train_loss' in metrics_dict:
            plt.plot(metrics_dict['train_loss'], label='Train')
        if 'val_loss' in metrics_dict:
            plt.plot(metrics_dict['val_loss'], label='Val')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot mIoU
        plt.subplot(1, 3, 2)
        if 'train_iou_mean' in metrics_dict:
            plt.plot(metrics_dict['train_iou_mean'], label='Train')
        if 'val_iou_mean' in metrics_dict:
            plt.plot(metrics_dict['val_iou_mean'], label='Val')
        plt.title(f'{model_name} - mIoU')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot F1
        plt.subplot(1, 3, 3)
        if 'train_f1_mean' in metrics_dict:
            plt.plot(metrics_dict['train_f1_mean'], label='Train')
        if 'val_f1_mean' in metrics_dict:
            plt.plot(metrics_dict['val_f1_mean'], label='Val')
        plt.title(f'{model_name} - F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'history_{model_name}.png')
        plt.savefig(save_path, dpi=200)
        print(f"Saved individual plot to {save_path}")
        plt.close()

def main():
    history = load_history()
    if history:
        print(f"Loaded history for models: {list(history.keys())}")
        plot_comparison(history)
        plot_individual_models(history)
    else:
        print("Failed to load history.")

if __name__ == "__main__":
    main()
