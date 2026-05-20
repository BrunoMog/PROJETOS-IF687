import json
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import Subset
from torchvision.transforms import v2
from ablation_utils import set_seed, CNNParameters, run_ablation_variant, split_train_val_dataset, RepeatChannels

# Config
ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "../datasets"
RESULTS_FILE = ROOT / "ablation_study_results.json"
MODEL_DIR = ROOT / "models"
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[ablation] device={device}", flush=True)

# Prepare MNIST
print('[ablation] Loading MNIST...', flush=True)
transform_mnist = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
training_data = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform_mnist)
test_data = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=transform_mnist)
print('[ablation] MNIST loaded', flush=True)
print('[ablation] Splitting train/val...', flush=True)
train_data, val_data = split_train_val_dataset(training_data, val_size=1.0/6.0)
print(f"[ablation] train/val sizes: {len(train_data)}/{len(val_data)}", flush=True)

# Quick mode: reduce dataset size for fast sanity-checks on CPU
QUICK = True
if QUICK:
    small_train_n = 2000
    small_val_n = 500
    try:
        train_indices = train_data.indices if hasattr(train_data, 'indices') else list(range(len(train_data)))
        val_indices = val_data.indices if hasattr(val_data, 'indices') else list(range(len(val_data)))
        train_data = Subset(train_data.dataset if hasattr(train_data, 'dataset') else train_data, train_indices[:small_train_n])
        val_data = Subset(val_data.dataset if hasattr(val_data, 'dataset') else val_data, val_indices[:small_val_n])
        print(f"[ablation] QUICK mode: reduced train/val sizes to {len(train_data)}/{len(val_data)}", flush=True)
    except Exception as e:
        print(f"[ablation] QUICK mode subset failed: {e}", flush=True)

# Common hyperparams
batch_size = 64
num_epochs = 1  # quick sanity-check (reduced)
patience = 1
lr = 1e-3

image_h = training_data[0][0].shape[1]
image_w = training_data[0][0].shape[2]

# Base params (from best found example, simplified)
base_out_channels = [8, 32]
base_fc = [128]

# Variant A: with dropout
params_with = CNNParameters(
    num_conv_layers=2,
    stride_conv_layers=1,
    input_channels=1,
    out_channels_list=base_out_channels,
    use_pooling=True,
    padding_conv_layer=1,
    dilatation=1,
    bias=True,
    kernel_size=3,
    kernel_stride=1,
    kernel_padding=0,
    kernel_dilatation=1,
    activation_function_conv='leaky_relu',
    fully_connected_layers_list=base_fc,
    activation_function_fully_connected='relu',
    dropout_rate_fully_connected=0.3,
    dropout_rate_conv=0.25,
    output_size_fully_connected=10,
    image_size_height=image_h,
    image_size_width=image_w,
)

# Variant B: without dropout
params_without = CNNParameters(
    num_conv_layers=2,
    stride_conv_layers=1,
    input_channels=1,
    out_channels_list=base_out_channels,
    use_pooling=True,
    padding_conv_layer=1,
    dilatation=1,
    bias=True,
    kernel_size=3,
    kernel_stride=1,
    kernel_padding=0,
    kernel_dilatation=1,
    activation_function_conv='leaky_relu',
    fully_connected_layers_list=base_fc,
    activation_function_fully_connected='relu',
    dropout_rate_fully_connected=0.0,
    dropout_rate_conv=0.0,
    output_size_fully_connected=10,
    image_size_height=image_h,
    image_size_width=image_w,
)

results = {}
print('[ablation] Running variant: with_dropout', flush=True)
results['with_dropout'], hist_with = run_ablation_variant(params_with, train_data, test_data, val_data, batch_size, num_epochs, patience, lr, str(MODEL_DIR / 'best_with.pth'), device, verbose=True)
print('[ablation] Finished variant: with_dropout', flush=True)
print('[ablation] Running variant: without_dropout', flush=True)
results['without_dropout'], hist_without = run_ablation_variant(params_without, train_data, test_data, val_data, batch_size, num_epochs, patience, lr, str(MODEL_DIR / 'best_without.pth'), device, verbose=True)
print('[ablation] Finished variant: without_dropout', flush=True)

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print('Results saved to', RESULTS_FILE)
print('With dropout:', results['with_dropout'])
print('Without dropout:', results['without_dropout'])
