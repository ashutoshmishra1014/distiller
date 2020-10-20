import torch
import torchvision.models as models
from torchvision.models import resnet18 as resnet18_v2
from distiller.models.dcase.asc import asc

checkpoint_path = '/home/mha/checkouts/trash/NNO_DISTILLER/usecase_pretrained_models/new_asc__200__.pth'
# model = models.resnet18(pretrained=True)
model_ckpt = torch.load(checkpoint_path)

# model_checkpoint = dict()
# model_checkpoint['state_dict'] = model.state_dict()
# model_checkpoint['epoch'] = 0
# model_checkpoint['arch'] = 'resnet18_v2'
# model_checkpoint['is_parallel'] = False
# model_checkpoint['dataset'] = 'imagenet_v2'

# torch.save(model_checkpoint, checkpoint_path)
# model_v2 = resnet18_v2(pretrained=False)
# model_v2.load_state_dict(torch.load(checkpoint_path)['state_dict'])

