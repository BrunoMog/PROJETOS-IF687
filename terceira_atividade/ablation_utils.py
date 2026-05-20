import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
from pathlib import Path

# Re-use RepeatChannels if present
try:
    from segunda_atividade.RepeatChannels import RepeatChannels
except Exception:
    try:
        from RepeatChannels import RepeatChannels
    except Exception:
        RepeatChannels = None


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


map_activation_functions = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
}
map_optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}

def get_activation_function(name):
    return map_activation_functions[name]()


class CNNParameters:
    def __init__(self, num_conv_layers, stride_conv_layers, input_channels, out_channels_list, use_pooling,
                 padding_conv_layer, dilatation, bias, kernel_size, kernel_stride, kernel_padding, kernel_dilatation,
                 activation_function_conv, fully_connected_layers_list, activation_function_fully_connected,
                 dropout_rate_fully_connected, dropout_rate_conv, output_size_fully_connected,
                 image_size_height, image_size_width):
        self.num_conv_layers = num_conv_layers
        self.stride_conv_layers = stride_conv_layers
        self.input_channels = input_channels
        self.out_channels_list = out_channels_list
        self.dropout_rate_conv = dropout_rate_conv
        self.padding_conv_layer = padding_conv_layer
        self.dilatation = dilatation
        self.bias = bias
        self.activation_function_conv = activation_function_conv

        self.use_pooling = use_pooling
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_padding = kernel_padding
        self.kernel_dilatation = kernel_dilatation

        self.fully_connected_layers_list = fully_connected_layers_list
        self.activation_function_fully_connected = activation_function_fully_connected
        self.dropout_rate_fully_connected = dropout_rate_fully_connected
        self.output_size_fully_connected = output_size_fully_connected

        self.image_size_height = image_size_height
        self.image_size_width = image_size_width
        self.image_size_height_output, self.image_size_width_output = self._calculate_output_size(image_size_height, image_size_width)
        self.flatten_size = self.image_size_height_output * self.image_size_width_output * self.out_channels_list[-1]

    def _calculate_output_size(self, input_size_height, input_size_width):
        output_size_height = input_size_height
        output_size_width = input_size_width
        for i in range(self.num_conv_layers):
            output_size_height = (output_size_height + 2 * self.padding_conv_layer - self.dilatation * (self.kernel_size - 1) - 1) // self.stride_conv_layers + 1
            output_size_width = (output_size_width + 2 * self.padding_conv_layer - self.dilatation * (self.kernel_size - 1) - 1) // self.stride_conv_layers + 1
            if self.use_pooling:
                output_size_height = (output_size_height + 2 * self.kernel_padding - self.kernel_dilatation * (self.kernel_size - 1) - 1) // self.kernel_stride + 1
                output_size_width = (output_size_width + 2 * self.kernel_padding - self.kernel_dilatation * (self.kernel_size - 1) - 1) // self.kernel_stride + 1
        return output_size_height, output_size_width


class CNN(nn.Module):
    def __init__(self, cnn_parameters: CNNParameters):
        super(CNN, self).__init__()
        self.cnn_parameters = cnn_parameters
        layers = []
        for i in range(self.cnn_parameters.num_conv_layers):
            layers.append(nn.Conv2d(in_channels=self.cnn_parameters.input_channels if i == 0 else self.cnn_parameters.out_channels_list[i-1],
                                    out_channels=self.cnn_parameters.out_channels_list[i],
                                    kernel_size=self.cnn_parameters.kernel_size,
                                    stride=self.cnn_parameters.stride_conv_layers,
                                    padding=self.cnn_parameters.padding_conv_layer,
                                    dilation=self.cnn_parameters.dilatation,
                                    bias=self.cnn_parameters.bias))
            layers.append(get_activation_function(self.cnn_parameters.activation_function_conv))
            if self.cnn_parameters.dropout_rate_conv > 0:
                layers.append(nn.Dropout2d(self.cnn_parameters.dropout_rate_conv))
            if self.cnn_parameters.use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=self.cnn_parameters.kernel_size,
                                           stride=self.cnn_parameters.kernel_stride,
                                           padding=self.cnn_parameters.kernel_padding,
                                           dilation=self.cnn_parameters.kernel_dilatation))
        self.conv_layers = nn.Sequential(*layers)

        fully_connected_layers = []
        input_size_fully_connected = cnn_parameters.flatten_size
        for output_size in self.cnn_parameters.fully_connected_layers_list:
            fully_connected_layers.append(nn.Linear(input_size_fully_connected, output_size))
            fully_connected_layers.append(get_activation_function(self.cnn_parameters.activation_function_fully_connected))
            if self.cnn_parameters.dropout_rate_fully_connected > 0:
                fully_connected_layers.append(nn.Dropout(self.cnn_parameters.dropout_rate_fully_connected))
            input_size_fully_connected = output_size
        fully_connected_layers.append(nn.Linear(input_size_fully_connected, self.cnn_parameters.output_size_fully_connected))
        self.fully_connected_layers = nn.Sequential(*fully_connected_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fully_connected_layers(x)
        return x


# Utilities

def evaluate_model(model, test_dataloader, metric_fn=metrics.accuracy_score, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y.extend(y_batch.detach().cpu().tolist())
            y_pred.extend(predicted.detach().cpu().tolist())
    metric_val = metric_fn(y, y_pred)
    return metric_val


def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, patience, metric_fn=None, device=None, verbose=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history_train_loss = []
    history_val_loss = []
    history_val_metric = []
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        history_train_loss.append(avg_train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        history_val_loss.append(avg_val_loss)
        if metric_fn is not None:
            metric_val = evaluate_model(model, val_dataloader, metric_fn=metric_fn, device=device)
            history_val_metric.append(metric_val)
        if verbose:
            try:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Metric: {history_val_metric[-1] if history_val_metric else 'N/A'}", flush=True)
            except Exception:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", flush=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    return history_train_loss, history_val_loss, history_val_metric


def create_model_and_dataloaders(cnn_parameters, train_data, test_data, val_data, batch_size):
    model = CNN(cnn_parameters)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return model, train_dataloader, test_dataloader, val_dataloader


def split_train_val_dataset(dataset, val_size, random_state=42):
    labels = np.asarray(dataset.targets)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=random_state, stratify=labels)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def plot_loss_graph(history_train_loss, history_val_loss, history_val_metric):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history_train_loss, label='Train Loss')
    ax1.plot(history_val_loss, label='Val Loss')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend()
    if history_val_metric:
        ax2.plot(history_val_metric, label='Val Accuracy', color='green')
        ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy'); ax2.legend()
    plt.tight_layout(); plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_ablation_variant(cnn_params, train_data, test_data, val_data, batch_size, num_epochs, patience, lr, save_path, device=None, verbose=False, compute_flops=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_dl, test_dl, val_dl = create_model_and_dataloaders(cnn_params, train_data, test_data, val_data, batch_size)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    history_train_loss, history_val_loss, history_val_metric = train_model(model, train_dl, val_dl, loss_fn, optimizer, num_epochs=num_epochs, patience=patience, metric_fn=metrics.accuracy_score, device=device, verbose=verbose)
    train_time = time.time() - start
    test_accuracy = evaluate_model(model, test_dl, metric_fn=metrics.accuracy_score, device=device)
    params = count_parameters(model)
    flops = None
    if compute_flops:
        try:
            flops = FlopCountAnalysis(model.cpu(), torch.randn(1, cnn_params.input_channels, cnn_params.image_size_height, cnn_params.image_size_width).cpu(),).total()
        except Exception:
            flops = None
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model_file = save_path
    torch.save(model.state_dict(), model_file)
    results = {
        'accuracy': float(test_accuracy),
        'params': int(params),
        'flops': float(flops) if flops is not None else None,
        'train_time_sec': float(train_time),
        'model_path': str(Path(model_file))
    }
    return results, (history_train_loss, history_val_loss, history_val_metric)
