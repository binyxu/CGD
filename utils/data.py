import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from utils.dataset.GTSRB import GTSRB
from utils.dataset.Tiny import TinyImageNet
from torchvision.datasets import CIFAR10, CIFAR100

DATASET_META = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny': 200,
    'gtsrb': 43,
}

TEMPLATE_META = {
    'cifar10': "a photo of a {}",
    'cifar100': "a photo of a {}",
    'tiny': "a photo of a {}",
    'gtsrb': "a centered photo of a {} traffic sign",
}

MAPPING_META = {
    'cifar10': None,
    'cifar100': None,
    'tiny': "utils/dataset/map_clsloc.txt",
    'gtsrb': None,
}

def load_dataset(name, root, transform):
    if name == 'cifar10':
        return CIFAR10(root=root, train=True, download=True, transform=transform)
    elif name == 'cifar100':
        return CIFAR100(root=root, train=True, download=True, transform=transform)
    elif name == 'tiny':
        return TinyImageNet(root=root, split="train", download=True, transform=transform)
    elif name == 'gtsrb':
        return GTSRB(data_root=f"{root}/gtsrb_bb", train=True, transform=transform)


def update_class2idx(class2idx, mapping_file_path):
    if not mapping_file_path:
        return class2idx
    
    # Create a new dictionary for the updated mapping
    updated_class2idx = {}

    # Open and read the mapping file
    with open(mapping_file_path, 'r') as f:
        for line in f:
            # Split the line by commas
            parts = line.strip().split(',')
            class_code = parts[0]  # The n021xxxx class code
            natural_language = parts[2]  # The natural language name

            # If the class code exists in the current class2idx, update the dictionary
            if class_code in class2idx:
                updated_class2idx[natural_language] = class2idx[class_code]

    return updated_class2idx

class PoisonedDataset(Dataset):
    def __init__(self, dataset, poisoned_data_dir, target=0, transform=None, name=None, pr=0.0):
        """
        Args:
            dataset (Dataset): The original dataset.
            poisoned_data_dir (str): Path to the directory containing poisoned data.
            target (int): The target label for poisoned samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self.poisoned_data_dir = poisoned_data_dir
        self.result_dict = torch.load(os.path.join(poisoned_data_dir, '../attack_result.pt'))['bd_train']['bd_data_container']['data_dict']
        self.target = target
        self.transform = transform

        # Dictionaries to store poisoned images and labels
        self.poisoned_images = {}
        self.poisoned_labels = {}

        self._load_poisoned_data()
        self.poisoned_indices = list(self.poisoned_images.keys())

        self.noise_indices = []

        if name == "bpp" or name == "wanet" or name == "inputaware":
            bs = 128
            self.poisoned_indices.sort()
            sublists = self.split_into_consecutive_sublists(self.poisoned_indices)
            for poison_list in sublists:
                self.noise_indices += poison_list[int(bs * pr):]
                

    def _load_poisoned_data(self):
        """
        Load poisoned images and labels from the poisoned_data_dir.
        Assumes that poisoned_data_dir has subdirectories named after class labels,
        and image filenames correspond to original dataset indices (e.g., '123.png').
        """
        if os.path.isdir(self.poisoned_data_dir):
            for class_label in os.listdir(self.poisoned_data_dir):
                class_dir = os.path.join(self.poisoned_data_dir, class_label)
                if os.path.isdir(class_dir):
                    for image_file in os.listdir(class_dir):
                        if image_file.endswith('.png'):
                            try:
                                # Extract the original index from the filename
                                original_index = int(os.path.splitext(image_file)[0])
                            except ValueError:
                                print(f"Filename {image_file} does not start with an integer index. Skipping.")
                                continue
                            
                            if original_index not in self.result_dict:
                                continue

                            poisoned_image_path = os.path.join(class_dir, image_file)
                            try:
                                with open(poisoned_image_path, 'rb') as f:
                                    poisoned_image = Image.open(f).convert('RGB')
                                    if self.transform:
                                        poisoned_image = self.transform(poisoned_image)
                                    else:
                                        poisoned_image = poisoned_image
                            except Exception as e:
                                print(f"Error loading image {poisoned_image_path}: {e}")
                                continue

                            # Store the poisoned image and target
                            self.poisoned_images[original_index] = poisoned_image
                            self.poisoned_labels[original_index] = self.result_dict[original_index]['other_info'][0]

    def split_into_consecutive_sublists(self, numbers):
        if not numbers:
            return []
        sublists = []
        current_sublist = [numbers[0]]
        for num in numbers[1:]:
            if num == current_sublist[-1] + 1:
                current_sublist.append(num)
            else:
                sublists.append(current_sublist)
                current_sublist = [num]
        sublists.append(current_sublist)
        return sublists

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item.

        Returns:
            tuple: (image, label) where label is the target for poisoned samples.
        """
        if index in self.poisoned_images:
            poisoned_image = self.poisoned_images[index]
            poisoned_label = self.poisoned_labels[index]
            return poisoned_image, poisoned_label
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

# Usage Example:
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    poisoned_data_dir = '/data/binyanxu/BackdoorBench_old/record/bad_cifar10_base/bd_train_dataset'  # Replace with your directory containing poisoned images
    dataset = CIFAR10(root='/data/datasets', train=True, download=True, transform=transform)
    dataset = PoisonedDataset(dataset=dataset, poisoned_data_dir=poisoned_data_dir)

    # DataLoader to handle batching
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Access poisoned indices
    print("Poisoned Indices:", dataset.poisoned_indices)
