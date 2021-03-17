from datasets.voc_dataset_custom import Stanford40
import os

def get_training_set(opt, transform):
    if opt.dataset == 'Stanford40':
        training_data = Stanford40(
            root=opt.dataset_path,
            transform=transform,
            is_test=False)
    return training_data

def get_validation_set(opt, transform):
    if opt.dataset == 'Stanford40':
        validation_data = Stanford40(
            root=opt.dataset_path,
            transform=transform,
            is_test=True)
    return validation_data