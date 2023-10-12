import os
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Optional, Tuple
from API.api_helper import fetch_images
from PIL import Image
import io
import base64

class GenericImageDataset(Dataset):
    def __init__(self, source, mode='external', transform: Optional[Callable] = None):
        """
        Args:
            source (str): Path to the local directory or name of the dataset in the external database.
            mode (str): 'local' for local directory or 'external' for external databases.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.mode = mode

        if mode == 'local':
            self.image_files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
            self.root_dir = source

        elif mode == 'external':
            self.dataset_name = source
            self.image_data = fetch_images(self.dataset_name)
        else:
            raise ValueError("Mode should be either 'local' or 'external'.")

    def __len__(self):
        if self.mode == 'local':
            return len(self.image_files)
        else:
            return len(self.image_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        if self.mode == 'local':
            image_path = os.path.join(self.root_dir, self.image_files[index])
            image_name = self.image_files[index]
            image = Image.open(image_path)

        else:
            image_info = self.image_data[index]
            image_name = image_info['name']
            image_bytes = base64.b64decode(image_info['data'])
            image = Image.open(io.BytesIO(image_bytes))

        if self.transform:
            image = self.transform(image)

        return image, image_name


# Example usage for local mode:
# dataset_local = GenericImageDataset(source='path_to_local_directory', mode='local', transform=transform)
# dataloader_local = DataLoader(dataset_local, batch_size=32, shuffle=True)

# Example usage for external mode:
# dataset_external = GenericImageDataset(source='your_dataset_name', mode='external', transform=transform)
# dataloader_external = DataLoader(dataset_external, batch_size=32, shuffle=True)


#
#
# class CustomLoader(Dataset):
#     def __init__(
#         self,
#         root: str,
#         transform: Optional[Callable] = None,
#     ) -> None:
#
#         super().__init__()
#
#         if self.train:
#             downloaded_list = self.train_list
#         else:
#             downloaded_list = self.test_list
#
#         self.data: Any = []
#         self.targets = []
#
#         # now load the picked numpy arrays
#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, "rb") as f:
#                 entry = pickle.load(f, encoding="latin1")
#                 self.data.append(entry["data"])
#                 if "labels" in entry:
#                     self.targets.extend(entry["labels"])
#                 else:
#                     self.targets.extend(entry["fine_labels"])
#
#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
#
#         self._load_meta()
#
#     def _load_meta(self) -> None:
#         path = os.path.join(self.root, self.base_folder, self.meta["filename"])
#         with open(path, "rb") as infile:
#             data = pickle.load(infile, encoding="latin1")
#             self.classes = data[self.meta["key"]]
#         self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     # def _check_integrity(self) -> bool:
#     #     for filename, md5 in self.train_list + self.test_list:
#     #         fpath = os.path.join(self.root, self.base_folder, filename)
#     #         if not check_integrity(fpath, md5):
#     #             return False
#     #     return True
#
#     # def download(self) -> None:
#     #     if self._check_integrity():
#     #         print("Files already downloaded and verified")
#     #         return
#     #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
#
#     # def extra_repr(self) -> str:
#     #     split = "Train" if self.train is True else "Test"
#     #     return f"Split: {split}"
#
