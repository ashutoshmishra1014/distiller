import os
import distiller
import torch.nn as nn
from distiller.models import register_user_model
import distiller.apputils.image_classifier as classifier


class MyModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model():
    return MyModel()


distiller.models.register_user_model(arch="MyModel", dataset="mnist", model=my_model)
model = distiller.models.create_model(pretrained=True, dataset="mnist", arch="MyModel")
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
    args.arch = "MyModel"
    args.dataset = "mnist"
    args.data = "./datasets/mnist/"
    args.pretrained = False
    args.lr = sgd_args[0]
    args.momentum = sgd_args[1]
    args.weight_decay = sgd_args[2]
    args.batch_size = 256
    args.epochs = epochs

args = classifier.init_classifier_compression_arg_parser()
init_jupyter_default_args(args)
config_learner_args(args, "MyModel", "mnist", "/datasets/mnist/", False, (0.1, 0.9, 1e-4) , 256, 1)
app = classifier.ClassifierCompressor(args, script_dir=os.path.dirname("."))

# Run the training loop
perf_scores_history = app.run_training_loop()
print(perf_scores_history)