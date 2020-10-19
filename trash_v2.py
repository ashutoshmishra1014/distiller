import os
import distiller
import torch.nn as nn
from distiller.models import register_user_model
import distiller.apputils.image_classifier as classifier

import torch
import torchvision.models as models


def my_model():
    return models.resnet18(pretrained=True)


distiller.models.register_user_model(arch="resnet_v3", dataset="imagenet_v2", model=my_model)
model = distiller.models.create_model(pretrained=True, dataset="imagenet_v2", arch="resnet_v3")
assert model is not None

def init_jupyter_default_args(args):
    args.output_dir = None
    args.evaluate = False
    args.seed = None
    args.deterministic = False
    args.cpu = True
    args.gpus = None
    args.load_serialized = False
    args.deprecated_resume = None
    args.resumed_checkpoint_path = None
    args.load_model_path = None
    args.reset_optimizer = False
    args.lr = args.momentum = args.weight_decay = 0.
    args.compress = None
    args.epochs = 0
    args.activation_stats = list()
    args.batch_size = 1
    args.workers = 1
    args.validation_split = 0.1
    args.effective_train_size = args.effective_valid_size = args.effective_test_size = 1.
    args.log_params_histograms = False
    args.print_freq = 1
    args.masks_sparsity = False
    args.display_confusion = False
    args.num_best_scores = 1
    args.name = ""


def config_learner_args(args, arch, dataset, dataset_path, pretrained, sgd_args, batch, epochs):
    args.arch = "resnet_v3"
    args.dataset = "imagenet_v2"
    args.data = "/media/mha/IIS-mha-Extern1/datasets/imagenet_1k/"
    args.pretrained = False
    args.lr = sgd_args[0]
    args.momentum = sgd_args[1]
    args.weight_decay = sgd_args[2]
    args.batch_size = 10
    args.epochs = epochs

args = classifier.init_classifier_compression_arg_parser()
init_jupyter_default_args(args)
config_learner_args(args, "resnet_v3", "imagenet_v2", "/media/mha/IIS-mha-Extern1/datasets/imagenet_1k", False, (0.1, 0.9, 1e-4) , 10, 10)
app = classifier.ClassifierCompressor(args, script_dir=os.path.dirname("./trash"))

# Run the training loop
perf_scores_history = app.run_training_loop()
print(perf_scores_history)

