from datasets.voc_dataset_custom import Stanford40
import os
import torchvision

def get_training_set(opt, transform):
    if opt.dataset == 'Stanford40':
        training_data = Stanford40(
            root=opt.dataset_path,
            transform=transform,
            is_test=False)
    
    if opt.dataset == 'Mask':
        training_data = torchvision.datasets.ImageFolder(
		f"{opt.dataset_path}/Train", transform=transform)
    return training_data

def get_validation_set(opt, transform):
    if opt.dataset == 'Stanford40':
        validation_data = Stanford40(
            root=opt.dataset_path,
            transform=transform,
            is_test=True)
    if opt.dataset == 'Mask':
        validation_data = torchvision.datasets.ImageFolder(
		f"{opt.dataset_path}/Validation", transform=transform)


    return validation_data