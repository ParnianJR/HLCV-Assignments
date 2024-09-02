from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.cnn_trainer import CNNTrainer
from src.models.cnn.model import ConvNet
from src.models.cnn.metric import TopKAccuracy

q1_experiment = dict(
    name = 'CIFAR10_CNN',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.Identity,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)


#########  TODO #####################################################
#  You would need to create the following config dictionaries       #
#  to use them for different parts of Q2 and Q3.                    #
#  Feel free to define more config files and dictionaries if needed.#
#  But make sure you have a separate config for every question so   #
#  that we can use them for grading the assignment.                 #
#####################################################################
q2a_normalization_experiment = dict(
    name = 'CIFAR10_CNN_q2a',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)

q2c_earlystop_experiment = dict(
    name = 'CIFAR10_CNN_q2c',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 4,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)



q3a_aug1_experiment = dict(
    name = 'CIFAR10_CNN_q3a_aug1',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10_Flip_Rotate',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 25,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)



q3a_aug2_experiment = dict(
    name = 'CIFAR10_CNN_q3a_aug2',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10_Scale_Crop',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 25,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)


q3a_aug3_experiment = dict(
    name = 'CIFAR10_CNN_q3a_aug3',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10_Color_Blur',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 25,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)


# q3a_aug2_experiment = ()
# q3a_aug3_experiment = ()
# ...


q3b_p1_dropout_experiment = dict(
    name = 'CIFAR10_CNN_q3b_p1',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.1,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 25,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 10,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)


q3b_p2_dropout_experiment = dict(
    name = 'CIFAR10_CNN_q3b_p2',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.3,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 25,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 10,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)


q3b_p3_dropout_experiment = dict(
    name = 'CIFAR10_CNN_q3b_p3',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.9,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 25,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "min eval_loss",
        early_stop = 10,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)

# define more config dictionaries if needed...