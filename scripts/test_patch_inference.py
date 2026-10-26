import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision import transforms
from src.objective_2.model import (
    EfficientNetB0TumorClassifier,
    ResNet50TumorClassifier,
    DenseNet121TumorClassifier,
    VGG16TumorClassifier
)

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cpu')

models = {}

# 1. EfficientNet
eff_ckpt = torch.load('result/objective_2/checkpoints/best_model.pt', map_location=device, weights_only=False)
m_eff = EfficientNetB0TumorClassifier(pretrained=False)
m_eff.load_state_dict(eff_ckpt['model_state_dict'])
m_eff.eval()
models['EfficientNet-B0'] = m_eff

# 2. ResNet
res_ckpt = torch.load('result/objective_2/checkpoints/resnet/best_model.pt', map_location=device, weights_only=False)
m_res = ResNet50TumorClassifier(pretrained=False)
m_res.load_state_dict(res_ckpt['model_state_dict'])
m_res.eval()
models['ResNet-50'] = m_res

# 3. DenseNet
dense_ckpt = torch.load('result/objective_2/checkpoints/densenet/best_model.pt', map_location=device, weights_only=False)
m_dense = DenseNet121TumorClassifier(pretrained=False)
m_dense.load_state_dict(dense_ckpt['model_state_dict'])
m_dense.eval()
models['DenseNet-121'] = m_dense

# 4. VGG
vgg_ckpt = torch.load('result/objective_2/checkpoints/vgg/best_model.pt', map_location=device, weights_only=False)
m_vgg = VGG16TumorClassifier(pretrained=False)
m_vgg.load_state_dict(vgg_ckpt['model_state_dict'])
m_vgg.eval()
models['VGG-16'] = m_vgg

print('All 4 models loaded successfully!')

test_norm_files = list(Path('data/PATCHES/TEST/normal').glob('*.png'))[:10]
test_tum_files = list(Path('data/PATCHES/TEST/tumor').glob('*.png'))[:10]

print('\n=== CAMELYON16 TEST NORMAL PATCHES ===')
for f in test_norm_files:
    img = Image.open(f).convert('RGB')
    t = eval_tf(img).unsqueeze(0)
    scores = {}
    for name, m in models.items():
        with torch.no_grad():
            p = torch.sigmoid(m(t)).item()
        scores[name] = round(p, 4)
    print(f"{f.name[:30]}: {scores}")

print('\n=== CAMELYON16 TEST TUMOR PATCHES ===')
for f in test_tum_files:
    img = Image.open(f).convert('RGB')
    t = eval_tf(img).unsqueeze(0)
    scores = {}
    for name, m in models.items():
        with torch.no_grad():
            p = torch.sigmoid(m(t)).item()
        scores[name] = round(p, 4)
    print(f"{f.name[:30]}: {scores}")

# Also check external datasets
pcam_norm = list(Path('data/external_test/patchcamelyon/normal').glob('*.png'))[:5]
pcam_tum = list(Path('data/external_test/patchcamelyon/tumor').glob('*.png'))[:5]

print('\n=== EXTERNAL PATCHCAMELYON NORMAL PATCHES ===')
for f in pcam_norm:
    img = Image.open(f).convert('RGB')
    t = eval_tf(img).unsqueeze(0)
    scores = {}
    for name, m in models.items():
        with torch.no_grad():
            p = torch.sigmoid(m(t)).item()
        scores[name] = round(p, 4)
    print(f"{f.name[:30]}: {scores}")

print('\n=== EXTERNAL PATCHCAMELYON TUMOR PATCHES ===')
for f in pcam_tum:
    img = Image.open(f).convert('RGB')
    t = eval_tf(img).unsqueeze(0)
    scores = {}
    for name, m in models.items():
        with torch.no_grad():
            p = torch.sigmoid(m(t)).item()
        scores[name] = round(p, 4)
    print(f"{f.name[:30]}: {scores}")

# BreakHis
bhis_norm = list(Path('data/external_test/breakhis/benign').glob('*.png'))[:5]
bhis_tum = list(Path('data/external_test/breakhis/malignant').glob('*.png'))[:5]

print('\n=== EXTERNAL BREAKHIS BENIGN PATCHES ===')
for f in bhis_norm:
    img = Image.open(f).convert('RGB')
    t = eval_tf(img).unsqueeze(0)
    scores = {}
    for name, m in models.items():
        with torch.no_grad():
            p = torch.sigmoid(m(t)).item()
        scores[name] = round(p, 4)
    print(f"{f.name[:30]}: {scores}")

print('\n=== EXTERNAL BREAKHIS MALIGNANT PATCHES ===')
for f in bhis_tum:
    img = Image.open(f).convert('RGB')
    t = eval_tf(img).unsqueeze(0)
    scores = {}
    for name, m in models.items():
        with torch.no_grad():
            p = torch.sigmoid(m(t)).item()
        scores[name] = round(p, 4)
    print(f"{f.name[:30]}: {scores}")
