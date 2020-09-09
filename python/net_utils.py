import os
import torch as th

from models import allcnn, mnistconv, mnistfc
from optimizers.entropy_sgd import EntropySGD
from optimizers.radam import RAdam
from torch import optim


def define_optimization(opt):
    """ Defines model and optimizer
        Model is randomly initialized or a checkpoint
        is loaded

    Args:
        opt (dict) -- experiment configurations

    Returns:
        model and optimizer objects
    """

    # define model
    if opt['m'] == 'mnistfc':
        model = mnistfc(opt)
        opt['dataset'] = 'mnist'
    elif opt['m'] == 'mnistconv':
        model = mnistconv(opt)
        opt['dataset'] = 'mnist'
    elif opt['m'] == 'allcnn':
        model = allcnn(opt)
        opt['dataset'] = 'cifar10'
    else:
        raise Exception("Invalid model name!")

    # print active layers
    require_grad = 0
    for param in model.parameters():
        if param.requires_grad:
            require_grad += 1

    print(f"Number of active layers: {require_grad}")

    # define optimizer
    if opt["optim"] == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt["lr"],
            momentum=opt["momentum"],
            nesterov=bool(opt["nesterov"]),
            weight_decay=opt["wd"],
        )
    elif opt["optim"] == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt["lr"],
            weight_decay=opt["wd"],
            amsgrad=False,
        )
    elif opt["optim"] == "entropy_sgd":
        optimizer = EntropySGD(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            config=dict(lr=opt['lr'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=opt['wd'],
                        L=opt['L'],
                        eps=opt['noise'],
                        g0=opt['gamma'],
                        g1=opt['scoping'])
        )
    elif opt["optim"] == "radam":
        optimizer = RAdam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt["lr"],
            weight_decay=0,
        )

    # load checkpoint when applicable
    if opt["resume"]:

        name = "{}_{}/data_{}/session_{}/".format(opt["m"],
                                                  opt["optim"],
                                                  opt["dataset"],
                                                  opt["session"])
        folder = opt["save_dir"] + name

        model_name = "{}_{}_{}_s{}_e{}.pth".format(opt["m"],
                                                   opt["optim"],
                                                   opt["dataset"],
                                                   opt["session"],
                                                   opt["checkepoch"])

        load_name = os.path.join(folder, model_name)

        print("loading checkpoint %s" % (load_name))
        checkpoint = th.load(load_name)
        opt["session"] = checkpoint["session"]
        opt["start_epoch"] = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer
