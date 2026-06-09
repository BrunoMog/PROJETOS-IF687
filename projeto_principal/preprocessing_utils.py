try:
    import torch
    import torchvision.transforms.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class DummyTensor:
        pass
    class DummyNamespace:
        Tensor = DummyTensor
    torch = DummyNamespace()

from PIL import Image, ImageOps
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    class DummyNP:
        pass
    np = DummyNP()

class AspectPreservingResizeAndPad:
    """
    Redimensiona uma imagem mantendo a proporção (aspect ratio) original com base
    no maior lado, e preenche (pad) os lados menores com uma cor de preenchimento
    para gerar uma imagem final quadrada (target_size x target_size).
    
    Suporta tanto PIL Images quanto PyTorch Tensors (e tv_tensors).
    """
    def __init__(self, target_size, fill=255):
        """
        Args:
            target_size (int): Tamanho final desejado da imagem (largura e altura).
            fill (int, tuple ou list): Cor de preenchimento das bordas.
                                       Padrão: 255 (branco para combinar com lâminas).
        """
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            raise TypeError(f"A imagem deve ser PIL Image ou torch Tensor. Tipo recebido: {type(img)}")

        # Calcula o novo tamanho mantendo a proporção original
        aspect_ratio = w / h
        if w > h:
            new_w = self.target_size
            new_h = int(round(self.target_size / aspect_ratio))
        else:
            new_h = self.target_size
            new_w = int(round(self.target_size * aspect_ratio))

        # Redimensiona a imagem
        if isinstance(img, Image.Image):
            resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            resized_img = F.resize(resized_img_tensor_wrapper(img), [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR)

        # Calcula o preenchimento necessário para centralizar a imagem
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        # Aplica o preenchimento (padding)
        if isinstance(img, Image.Image):
            if isinstance(self.fill, (int, float)):
                fill_color = (int(self.fill), int(self.fill), int(self.fill))
            else:
                fill_color = self.fill
            
            padded_img = Image.new(img.mode, (self.target_size, self.target_size), fill_color)
            padded_img.paste(resized_img, (pad_left, pad_top))
            return padded_img
        else:
            padding = [pad_left, pad_top, pad_right, pad_bottom]
            return F.pad(resized_img, padding, fill=self.fill, padding_mode="constant")

    def __repr__(self):
        return f"{self.__class__.__name__}(target_size={self.target_size}, fill={self.fill})"


def resized_img_tensor_wrapper(img):
    # Garante que seja um tensor ou tv_tensor válido
    return img


class ApplyCLAHE:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para realçar
    características do núcleo e do citoplasma nos glóbulos brancos.
    Equaliza o canal de luminância (L do espaço de cores LAB).
    
    Suporta tanto PIL Images quanto PyTorch Tensors.
    Caso o OpenCV (cv2) não esteja instalado, faz fallback para ImageOps.equalize da PIL.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit (float): Limite de contraste no CLAHE.
            tile_grid_size (tuple): Tamanho da grade para equalização local.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._cv2_available = True
        try:
            import cv2
        except ImportError:
            self._cv2_available = False

    def __call__(self, img):
        if self._cv2_available:
            import cv2
            if isinstance(img, Image.Image):
                # Conversão para numpy
                img_np = np.array(img)
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                return Image.fromarray(enhanced)
            elif isinstance(img, torch.Tensor):
                device = img.device
                dtype = img.dtype
                
                # Transpõe [C, H, W] -> [H, W, C] se for 3 canais
                img_np = img.detach().cpu().numpy()
                is_channels_first = img_np.ndim == 3 and img_np.shape[0] in [1, 3]
                if is_channels_first:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Converte floats [0.0, 1.0] para uint8 [0, 255] se necessário
                is_float = np.issubdtype(img_np.dtype, np.floating)
                if is_float:
                    # Se max <= 1.0 supomos normalizado, senão supomos que já está em escala 0-255
                    if img_np.max() <= 1.01:
                        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = img_np.clip(0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                
                # Aplica CLAHE
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                
                # Converte de volta para Tensor
                enhanced_tensor = torch.from_numpy(enhanced).to(device)
                if is_channels_first:
                    enhanced_tensor = enhanced_tensor.permute(2, 0, 1)
                
                if is_float:
                    enhanced_tensor = enhanced_tensor.to(dtype) / 255.0
                else:
                    enhanced_tensor = enhanced_tensor.to(dtype)
                
                return enhanced_tensor
        else:
            # Fallback para equalização global padrão PIL se cv2 não estiver instalado
            if isinstance(img, Image.Image):
                return ImageOps.equalize(img)
            elif isinstance(img, torch.Tensor):
                # Conversão para PIL temporária para fazer o fallback
                device = img.device
                dtype = img.dtype
                img_np = img.detach().cpu().numpy()
                is_channels_first = img_np.ndim == 3 and img_np.shape[0] in [1, 3]
                if is_channels_first:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Converte float para uint8
                if np.issubdtype(img_np.dtype, np.floating):
                    if img_np.max() <= 1.01:
                        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = img_np.clip(0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                
                pil_img = Image.fromarray(img_np)
                equalized_pil = ImageOps.equalize(pil_img)
                equalized_np = np.array(equalized_pil)
                
                enhanced_tensor = torch.from_numpy(equalized_np).to(device)
                if is_channels_first:
                    enhanced_tensor = enhanced_tensor.permute(2, 0, 1)
                
                if np.issubdtype(dtype, np.floating):
                    enhanced_tensor = enhanced_tensor.to(dtype) / 255.0
                else:
                    enhanced_tensor = enhanced_tensor.to(dtype)
                return enhanced_tensor
            else:
                raise TypeError(f"A imagem deve ser PIL Image ou torch Tensor. Tipo recebido: {type(img)}")

    def __repr__(self):
        return f"{self.__class__.__name__}(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"
