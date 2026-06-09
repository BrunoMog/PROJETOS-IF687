import os
from PIL import Image, ImageDraw, ImageOps
from preprocessing_utils import AspectPreservingResizeAndPad, ApplyCLAHE

def create_mock_cell():
    """Gera uma imagem de célula fictícia de 800x400 pixels para testar distorção."""
    # Fundo rosa claro (típico de esfregaço de sangue)
    img = Image.new("RGB", (800, 400), (245, 230, 235))
    draw = ImageDraw.Draw(img)
    
    # Desenha o citoplasma da célula (círculo azul-claro) no centro
    # Centro: (400, 200), raio: 120
    draw.ellipse([280, 80, 520, 320], fill=(180, 200, 255), outline=(150, 170, 220), width=3)
    
    # Desenha o núcleo (círculo roxo escuro)
    draw.ellipse([340, 140, 460, 260], fill=(90, 30, 120))
    draw.ellipse([370, 160, 450, 240], fill=(110, 40, 150))
    
    return img

def main():
    print("Gerando imagem de célula simulada (800x400)...")
    original = create_mock_cell()
    
    # 1. Redimensionamento Direto (Baseline) - causa distorção
    print("Aplicando redimensionamento direto (224x224)...")
    direct_resized = original.resize((224, 224), Image.Resampling.BILINEAR)
    
    # 2. Pré-processamento Sofisticado (Preservação de Aspect Ratio + Padding)
    print("Aplicando AspectPreservingResizeAndPad (224x224)...")
    padding_resizer = AspectPreservingResizeAndPad(target_size=224, fill=255) # Preenchimento branco
    padded_resized = padding_resizer(original)
    
    # 3. CLAHE / Equalização
    print("Aplicando realce de contraste (ApplyCLAHE)...")
    clahe_applier = ApplyCLAHE(clip_limit=2.0)
    enhanced = clahe_applier(padded_resized)
    
    # Criar uma imagem combinada para comparação visual lado a lado
    # Largura total: 224 * 3 = 672, Altura: 224
    # Adicionamos a imagem original redimensionada mantendo a proporção em outra escala para contexto, ou apenas as 3 de 224x224.
    # Vamos mostrar: [Redimensionamento Direto (Distorcido)] [Aspect Preserving + Pad] [Aspect Preserving + Pad + Equalized]
    comp_img = Image.new("RGB", (672, 224))
    comp_img.paste(direct_resized, (0, 0))
    comp_img.paste(padded_resized, (224, 0))
    comp_img.paste(enhanced, (448, 0))
    
    # Desenhar rótulos nas imagens para clareza
    draw = ImageDraw.Draw(comp_img)
    draw.text((10, 10), "1. Direto (Distorcido)", fill=(0, 0, 0))
    draw.text((234, 10), "2. Com Padding (Preservado)", fill=(0, 0, 0))
    draw.text((458, 10), "3. Com Padding + Realce", fill=(0, 0, 0))
    
    # Salvar nos destinos
    output_path_proj = "/home/alisson/Músicas/PROJETOS-IF687/projeto_principal/mock_comparison.png"
    output_path_art = "/home/alisson/.gemini/antigravity/brain/aa6c6f5b-075d-4898-8b9a-dabc780bd930/mock_comparison.png"
    
    comp_img.save(output_path_proj)
    print(f"Salvo em: {output_path_proj}")
    
    # Salvar na pasta de artefatos
    os.makedirs(os.path.dirname(output_path_art), exist_ok=True)
    comp_img.save(output_path_art)
    print(f"Salvo nos artefatos em: {output_path_art}")

if __name__ == "__main__":
    main()
