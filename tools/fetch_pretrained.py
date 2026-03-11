import torch
import torchvision.models as models
import os

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(out_dir, exist_ok=True)

print('Downloading and saving ResNet18 weights...')
res = models.resnet18(pretrained=True)
res_path = os.path.join(out_dir, 'resnet18_pretrained.pt')
torch.save(res.state_dict(), res_path)
print('Saved', res_path)

print('Downloading and saving MobileNetV2 weights...')
mb = models.mobilenet_v2(pretrained=True)
mb_path = os.path.join(out_dir, 'mobilenet_v2_pretrained.pt')
torch.save(mb.state_dict(), mb_path)
print('Saved', mb_path)
