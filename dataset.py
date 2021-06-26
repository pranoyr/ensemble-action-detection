from datasets.stanford import Stanford40
import os
import torchvision


def get_training_set(opt, transform):
    training_data = []
    if opt.dataset == 'Stanford40':
        training_data.append(Stanford40(
            root=opt.dataset_path,
            transform=transform,
            is_test=False))
    
    if opt.dataset == 'Mask':
        training_data.append(torchvision.datasets.ImageFolder(
		f"{opt.dataset_path}/Train", transform=transform))

    return training_data

def get_validation_set(opt, transform):
    validation_data = []
    if opt.dataset == 'Stanford40':
        validation_data.append(Stanford40(
            root=opt.dataset_path,
            transform=transform,
            is_test=True))
    if opt.dataset == 'Mask':
        validation_data.append(torchvision.datasets.ImageFolder(
		f"{opt.dataset_path}/Validation", transform=transform))


    return validation_data