import torch
import torchvision.models as models
# from torchvision.models import resnet18 as resnet18_v2
# from distiller.models.dcase.asc import asc
from distiller.models.mri.unet import unet

checkpoint_path = '/home/mha/checkouts/trash/NNO_DISTILLER/usecase_pretrained_models/unet_distiller.pth'
# # model = models.resnet18(pretrained=True)
# state_dict = torch.load(checkpoint_path)
# print(state_dict)


# model_checkpoint = dict()
# model_checkpoint['state_dict'] = state_dict
# model_checkpoint['epoch'] = 0
# model_checkpoint['arch'] = 'unet'
# model_checkpoint['is_parallel'] = False
# model_checkpoint['dataset'] = 'mri'

# torch.save(model_checkpoint, '/home/mha/checkouts/trash/NNO_DISTILLER/usecase_pretrained_models/unet_distiller.pth')
model = unet(pretrained=False)
# model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
print(model.state_dict())

