import json

def main():
    notebook_path = '/home/alisson/Músicas/PROJETOS-IF687/projeto_principal/main_executado.ipynb'
    print(f"Lendo notebook {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified_cells = 0
    
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        
        source = cell.get('source', [])
        source_str = ''.join(source)
        
        # 1. Modificar DIFF_THRESHOLD na célula 4
        if 'DIFF_THRESHOLD = 0.1' in source_str:
            new_source = []
            for line in source:
                if 'DIFF_THRESHOLD = 0.1' in line:
                    new_source.append(line.replace('DIFF_THRESHOLD = 0.1', 'DIFF_THRESHOLD = 0.0'))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print(f"-> Célula {i} modificada: DIFF_THRESHOLD alterado para 0.0")
            modified_cells += 1
            
        # 2. Modificar ResNet18 (célula 66)
        elif 'models.resnet18' in source_str and 'param.requires_grad = False' in source_str:
            new_source = [
                "model_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)\n",
                "\n",
                "model_resnet.fc = nn.Linear(model_resnet.fc.in_features, len(CLASSES_DO_PROJETO))\n",
                "\n",
                "# Desbloquear todas as camadas para fine-tuning total\n",
                "for param in model_resnet.parameters():\n",
                "    param.requires_grad = True\n",
                "\n",
                "loss_fn_resnet = nn.CrossEntropyLoss()\n",
                "# Taxas de aprendizado diferenciais: 1e-5 para camadas base convolucionais, 1e-3 para fc final\n",
                "optimizer_resnet = torch.optim.Adam([\n",
                "    {'params': [p for name, p in model_resnet.named_parameters() if 'fc' not in name], 'lr': 1e-5},\n",
                "    {'params': model_resnet.fc.parameters(), 'lr': 1e-3}\n",
                "])\n"
            ]
            cell['source'] = new_source
            print(f"-> Célula {i} modificada: ResNet18 atualizada para fine-tuning com taxas de aprendizado diferenciais.")
            modified_cells += 1
            
        # 3. Modificar MobileNetV3 (célula 79)
        elif 'models.mobilenet_v3_large' in source_str and 'param.requires_grad = False' in source_str:
            new_source = [
                "model_mobile_net_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)\n",
                "\n",
                "model_mobile_net_v3_large.classifier[3] = nn.Linear(model_mobile_net_v3_large.classifier[3].in_features, len(CLASSES_DO_PROJETO))\n",
                "\n",
                "# Desbloquear todas as camadas para fine-tuning total\n",
                "for param in model_mobile_net_v3_large.parameters():\n",
                "    param.requires_grad = True\n",
                "\n",
                "loss_fn_mobile_net_v3_large = nn.CrossEntropyLoss()\n",
                "# Taxas de aprendizado diferenciais: 1e-5 para a base convolucional, 1e-3 para o classificador final\n",
                "optimizer_mobile_net_v3_large = torch.optim.Adam([\n",
                "    {'params': [p for name, p in model_mobile_net_v3_large.named_parameters() if 'classifier.3' not in name], 'lr': 1e-5},\n",
                "    {'params': model_mobile_net_v3_large.classifier[3].parameters(), 'lr': 1e-3}\n",
                "])\n"
            ]
            cell['source'] = new_source
            print(f"-> Célula {i} modificada: MobileNetV3 Large atualizada para fine-tuning com taxas de aprendizado diferenciais.")
            modified_cells += 1
            
        # 4. Modificar EfficientNet-B0 (célula 92)
        elif 'models.efficientnet_b0' in source_str and 'param.requires_grad = False' in source_str:
            new_source = [
                "model_efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)\n",
                "\n",
                "model_efficientnet_b0.classifier[1] = nn.Linear(model_efficientnet_b0.classifier[1].in_features, len(CLASSES_DO_PROJETO))\n",
                "\n",
                "# Desbloquear todas as camadas para fine-tuning total\n",
                "for param in model_efficientnet_b0.parameters():\n",
                "    param.requires_grad = True\n",
                "\n",
                "loss_fn_efficientnet_b0 = nn.CrossEntropyLoss()\n",
                "# Taxas de aprendizado diferenciais: 1e-5 para a base convolucional, 1e-3 para o classificador final\n",
                "optimizer_efficientnet_b0 = torch.optim.Adam([\n",
                "    {'params': [p for name, p in model_efficientnet_b0.named_parameters() if 'classifier.1' not in name], 'lr': 1e-5},\n",
                "    {'params': model_efficientnet_b0.classifier[1].parameters(), 'lr': 1e-3}\n",
                "])\n"
            ]
            cell['source'] = new_source
            print(f"-> Célula {i} modificada: EfficientNet-B0 atualizada para fine-tuning com taxas de aprendizado diferenciais.")
            modified_cells += 1
            
    if modified_cells > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Notebook {notebook_path} atualizado com sucesso. Total de células modificadas: {modified_cells}")
    else:
        print("⚠ Nenhuma célula correspondente aos critérios foi encontrada para modificação.")

if __name__ == '__main__':
    main()
