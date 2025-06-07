"""
Final complete GCN script with all metrics on single graph
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set seeds
np.random.seed(2)
torch.manual_seed(2)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from stage_5_adapted import Method_GCN_Adapted, Evaluate_Metrics_Adapted
from Dataset_Loader_Node_Classification import Dataset_Loader


def plot_all_metrics_single_graph(curves, dataset_name, save_path):
    """Plot all metrics on a single comprehensive graph"""

    epochs = curves['epochs']

    # Create figure with 3 subplots (Train, Val, Test)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'GCN Performance Metrics - {dataset_name.upper()} Dataset', fontsize=16)

    # Define colors for each metric
    colors = {
        'Accuracy': 'blue',
        'F1 micro': 'green',
        'F1 macro': 'red',
        'F1 weighted': 'purple',
        'Precision micro': 'orange',
        'Precision macro': 'brown',
        'Precision weighted': 'pink',
        'Recall micro': 'gray',
        'Recall macro': 'olive',
        'Recall weighted': 'cyan'
    }

    # Line styles for variety
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted',
               'Precision micro', 'Precision macro', 'Precision weighted',
               'Recall micro', 'Recall macro', 'Recall weighted']

    # Plot for each split
    for idx, (ax, split) in enumerate(zip(axes, ['train', 'val', 'test'])):
        lines_plotted = False

        for i, metric in enumerate(metrics):
            key = f'{split}_{metric}'

            if key in curves and len(curves[key]) > 0:
                ax.plot(epochs, curves[key],
                       color=colors[metric],
                       linestyle=line_styles[i % len(line_styles)],
                       linewidth=2,
                       label=metric,
                       alpha=0.8)
                lines_plotted = True

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        ax.set_title(f'{split.capitalize()} Metrics')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        if lines_plotted and idx == 0:  # Only show legend on first subplot
            ax.legend(loc='lower right', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_name}_all_metrics_combined.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create a second figure showing final performance comparison
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Get final values for all metrics
    final_values = {'Train': [], 'Validation': [], 'Test': []}
    metric_names = []

    for metric in metrics:
        if f'test_{metric}' in curves and len(curves[f'test_{metric}']) > 0:
            metric_names.append(metric)
            final_values['Train'].append(curves[f'train_{metric}'][-1] if f'train_{metric}' in curves else 0)
            final_values['Validation'].append(curves[f'val_{metric}'][-1] if f'val_{metric}' in curves else 0)
            final_values['Test'].append(curves[f'test_{metric}'][-1] if f'test_{metric}' in curves else 0)

    if metric_names:
        x = np.arange(len(metric_names))
        width = 0.25

        bars1 = ax2.bar(x - width, final_values['Train'], width, label='Train', alpha=0.8)
        bars2 = ax2.bar(x, final_values['Validation'], width, label='Validation', alpha=0.8)
        bars3 = ax2.bar(x + width, final_values['Test'], width, label='Test', alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Final Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.1)

        # Add target accuracy line if Accuracy is present
        if 'Accuracy' in metric_names:
            acc_idx = metric_names.index('Accuracy')
            ax2.axhline(y=0.801, color='red', linestyle='--', alpha=0.5, label='Target (80.1%)')
            ax2.axhspan(0.796, 0.806, alpha=0.2, color='red', label='Target Range')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_name}_final_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create loss plot separately
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(epochs, curves['train_loss'], 'b-', label='Train', linewidth=2)
    ax3.plot(epochs, curves['val_loss'], 'g--', label='Validation', linewidth=2)
    ax3.plot(epochs, curves['test_loss'], 'r:', label='Test', linewidth=2)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_title(f'Loss Curves - {dataset_name.upper()}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_name}_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Configuration
    DATASET_NAME = 'cora'

    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(code_dir)

    data_dir = os.path.join(project_root, 'data', 'stage_5_data', DATASET_NAME)
    results_dir = os.path.join(project_root, 'result', 'stage_5_result')
    os.makedirs(results_dir, exist_ok=True)

    print(f"Running GCN on {DATASET_NAME} dataset")
    print(f"Target: 80.1% ± 0.5% accuracy\n")

    # Load data
    loader = Dataset_Loader(seed=2, dName=DATASET_NAME)
    loader.dataset_source_folder_path = data_dir
    loader.dataset_name = DATASET_NAME
    data = loader.load()

    # Initialize and run GCN
    gcn = Method_GCN_Adapted(f'GCN_{DATASET_NAME}', f'GCN on {DATASET_NAME}')
    gcn.data = data

    # Adjust hyperparameters for better performance
    gcn.hidden_dim = 16
    gcn.dropout_rate = 0.5
    gcn.learning_rate = 0.02  # Slightly higher learning rate
    gcn.weight_decay = 5e-4
    gcn.max_epoch = 300  # More epochs
    gcn.patience = 30  # More patience

    # Run training
    results = gcn.run()

    # Check if we have the metrics in curves
    print("\nChecking available metrics in curves...")
    metric_count = 0
    for key in results['curves'].keys():
        if key.startswith('test_') and key != 'test_loss':
            metric_count += 1
    print(f"Found {metric_count} test metrics in curves")

    # Plot results
    print("\nGenerating comprehensive plots...")
    plot_all_metrics_single_graph(results['curves'], DATASET_NAME, results_dir)

    # Print final test results
    print("\nFinal Test Results:")
    print("=" * 50)

    if results.get('test_metrics') is not None:
        for metric, value in results['test_metrics'].items():
            print(f"{metric}: {value:.4f}")

        # Check target accuracy
        test_acc = results['test_metrics']['Accuracy']
        print(f"\nTest Accuracy: {test_acc*100:.1f}%")
        if 79.6 <= test_acc*100 <= 80.6:
            print("✓ TARGET ACHIEVED! (80.1% ± 0.5%)")
        else:
            diff = test_acc*100 - 80.1
            print(f"✗ Target not achieved (80.1% ± 0.5%)")
            print(f"  Difference: {diff:+.1f}%")

    # Save detailed results
    with open(os.path.join(results_dir, f'{DATASET_NAME}_results.txt'), 'w') as f:
        f.write(f"GCN Results on {DATASET_NAME.upper()} Dataset\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Hidden dimension: {gcn.hidden_dim}\n")
        f.write(f"Dropout rate: {gcn.dropout_rate}\n")
        f.write(f"Learning rate: {gcn.learning_rate}\n")
        f.write(f"Weight decay: {gcn.weight_decay}\n")
        f.write(f"Max epochs: {gcn.max_epoch}\n")
        f.write(f"Patience: {gcn.patience}\n\n")

        if results.get('test_metrics') is not None:
            f.write("Final Test Metrics:\n")
            f.write("-" * 30 + "\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")

            test_acc = results['test_metrics']['Accuracy']
            f.write(f"\nAccuracy: {test_acc*100:.1f}%\n")
            f.write(f"Target: 80.1% ± 0.5%\n")
            f.write(f"Status: {'PASS' if 79.6 <= test_acc*100 <= 80.6 else 'FAIL'}\n")

        # Training summary
        curves = results['curves']
        f.write(f"\nTraining Summary:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total epochs trained: {len(curves['epochs'])}\n")

        # Best metrics
        if 'val_Accuracy' in curves:
            best_val_acc = max(curves['val_Accuracy'])
            best_epoch = curves['val_Accuracy'].index(best_val_acc)
            f.write(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n")
            f.write(f"Test accuracy at best val: {curves['test_Accuracy'][best_epoch]:.4f}\n")

    print(f"\nResults saved to: {results_dir}")

    # Provide recommendations if target not achieved
    if results.get('test_metrics') is not None:
        test_acc = results['test_metrics']['Accuracy']
        if not (79.6 <= test_acc*100 <= 80.6):
            print("\nRecommendations to achieve target:")
            if test_acc*100 < 79.6:
                print("- Try increasing learning rate to 0.02 or 0.03")
                print("- Train for more epochs (300-400)")
                print("- Try hidden_dim = 32")
            else:
                print("- Try decreasing learning rate to 0.005")
                print("- Increase dropout to 0.6")
                print("- Add more weight decay (1e-3)")


if __name__ == "__main__":
    main()