import torch
import torchvision.models as models
from torchvision.models import resnet18 as resnet18_v2

checkpoint_path = '/home/mha/checkouts/NNO_DISTILLER/usecase_pretrained_models/resnet18_distiller_v3.pth'
model = models.resnet18(pretrained=True)

model_checkpoint = dict()
model_checkpoint['state_dict'] = model.state_dict()
model_checkpoint['epoch'] = 0
model_checkpoint['arch'] = 'resnet18_v2'
model_checkpoint['is_parallel'] = False
model_checkpoint['dataset'] = 'imagenet_v2'

torch.save(model_checkpoint, checkpoint_path)
# model_v2 = resnet18_v2(pretrained=False)
# model_v2.load_state_dict(torch.load(checkpoint_path)['state_dict'])

