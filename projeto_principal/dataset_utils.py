import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class RaabinDataset(Dataset):
    """
    Dataset customizado para o banco de dados Raabin WBC.
    Carrega imagens sob demanda (lazy loading) para evitar problemas de memória.
    Compatível com multiprocessing graças ao armazenamento de caminhos.
    """
    def __init__(self, root_dir, class_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_list)}
        self.samples = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.class_to_idx:
                target_idx = self.class_to_idx[class_name]
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target_idx = self.samples[idx]
        
        # Abrir imagem sob demanda
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # Aplicar transformações
        if self.transform is not None:
            image = self.transform(image)

        # Retornar como tensor
        target = torch.tensor(target_idx, dtype=torch.long)
        return image, target
    
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]
