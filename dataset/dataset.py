import torch
import numpy as np
import os
import h5py


class Dataset(torch.utils.data.Dataset):
    def __init__(self, point_clouds, class_ids):
        self.point_clouds = torch.from_numpy(point_clouds).float()
        self.class_ids = torch.from_numpy(class_ids).long()

    def __len__(self):
        return np.shape(self.point_clouds)[0]

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.class_ids[idx]


def create_datasets_and_dataloaders(args):
    assert(os.path.exists(args.in_data_file))
    f = h5py.File(args.in_data_file, 'r')

    train_data = Dataset(f['train_point_clouds'][:], f['train_class_ids'][:])
    test_data = Dataset(f['test_point_clouds'][:], f['test_class_ids'][:])

    n_classes = np.amax(f['train_class_ids']) + 1
   # print('# classes: {:d}'.format(n_classes))

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=int(args.n_workers))

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.n_workers))

    return train_data, train_dataloader, test_data, test_dataloader, n_classes