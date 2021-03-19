import torch
import argparse

from distiller.models import create_model
from distiller.apputils.checkpoint import load_checkpoint
from distiller.apputils.data_loaders import classification_get_input_shape
from distiller import make_non_parallel_copy


parser = argparse.ArgumentParser(description='Distiller: convert checkpoint to torchscript module')
parser.add_argument('checkpoint', metavar='CHECKPOINT_PATH', help='path to checkpoint')
parser.add_argument('jitpath', metavar='JIT_PATH', help='save path torchscript module')

args = parser.parse_args()


def extract_checkpoint(checkpoint_path):
    model_items_dict = torch.load(checkpoint_path)
    
    return dict(
        (('arch', model_items_dict['arch']), 
            ('dataset', model_items_dict['dataset']), 
            ('parallel', model_items_dict['is_parallel'])
            )
        )
    
def get_model_from_checkpoint(checkpoint_path):
    model_arguments = extract_checkpoint(checkpoint_path)
    model = create_model(pretrained=False, **model_arguments)
    model = load_checkpoint(model, checkpoint_path, model_device=None, lean_checkpoint=True)[0]
    model = make_non_parallel_copy(model)
    model.dataset = model_arguments['dataset']
    return model.cuda()

def create_model_scriptmodule(model):
    dummy_input = torch.randn(classification_get_input_shape(model.dataset))
    dummy_input = dummy_input.cuda()
    traced_module = torch.jit.trace(model, dummy_input)
    return traced_module

def save_scriptmodule(scriptmodule, jitpath):
    assert scriptmodule is not None and jitpath is not None, "scriptmodule and jitpath required"
    torch.jit.save(scriptmodule, jitpath)
    

model = get_model_from_checkpoint(args.checkpoint)
traced_model = create_model_scriptmodule(model)
save_scriptmodule(traced_model, args.jitpath)



