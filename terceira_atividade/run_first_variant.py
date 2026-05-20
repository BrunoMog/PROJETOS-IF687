import json
from pathlib import Path
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset
from ablation_utils import set_seed, CNNParameters, run_ablation_variant, split_train_val_dataset

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "../datasets"
RESULTS_FILE = ROOT / "ablation_study_results.json"
MODEL_DIR = ROOT / "models"
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[first] device=', device, flush=True)

transform_mnist = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
training_data = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform_mnist)
test_data = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=transform_mnist)
train_data, val_data = split_train_val_dataset(training_data, val_size=1.0/6.0)

# QUICK subsets
train_data = Subset(train_data.dataset if hasattr(train_data,'dataset') else train_data, (train_data.indices if hasattr(train_data,'indices') else list(range(len(train_data))))[:2000])
val_data = Subset(val_data.dataset if hasattr(val_data,'dataset') else val_data, (val_data.indices if hasattr(val_data,'indices') else list(range(len(val_data))))[:500])

image_h = training_data[0][0].shape[1]
image_w = training_data[0][0].shape[2]

base_out_channels = [8, 32]
base_fc = [128]

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

results = {}
res, hist = run_ablation_variant(params_with, train_data, test_data, val_data, batch_size=64, num_epochs=1, patience=1, lr=1e-3, save_path=str(MODEL_DIR/'best_with.pth'), device=device, verbose=True, compute_flops=False)
results['with_dropout'] = res

# merge with existing results
if RESULTS_FILE.exists():
    try:
        old = json.loads(RESULTS_FILE.read_text())
        old.update(results)
        RESULTS_FILE.write_text(json.dumps(old, indent=2))
    except Exception:
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
else:
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

print('Saved results to', RESULTS_FILE, flush=True)
print('Results:', results, flush=True)
