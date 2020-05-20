import h5py
import torch
from torch.utils.data import Dataset


# torch.multiprocessing.set_start_method("spawn")


class H5DataReader:
    def __init__(self, path):

        ## If this crashes, try swmr=False
        self.mapping = {
            "F": h5py.File(f"{path}/Fgt.h5", "r", libver="latest", swmr=True),
            "matches": h5py.File(f"{path}/matches.h5", "r", libver="latest", swmr=True),
            "confidence": h5py.File(
                f"{path}/match_conf.h5", "r", libver="latest", swmr=True
            ),
        }

        self.path = path
        self.keys = self.__get_h5_keys(self.mapping["F"])
        self.num_keys = len(self.keys)

    def __get_h5_keys(self, file):
        return [key for key in file.keys()]

    def __load_h5_key(self, mapping, key):
        out = {k: v[key][()] for k, v in mapping.items()}

        return out

    def __getitem__(self, idx):
        return self.__load_h5_key(self.mapping, self.keys[idx]), self.keys[idx]

    def __len__(self):
        return len(self.keys)


class DummyH5Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.reader = None

    def __getitem__(self, idx):
        ## This is important to make hdf5 work with multiprocessing
        ## Opening the file in the constructor will lead to crashes
        if self.reader is None:
            self.reader = H5DataReader(self.path)

        data, name = self.reader.__getitem__(idx)

        ## Do something

    def __len__(self):
        return len(H5DataReader(self.path))
