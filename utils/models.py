import torch
import open_clip
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
from torch.optim import Adam
from utils.preact_resnet import PreActResNet18

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return idx, data  # Return index along with data


class KFoldClassifier:
    def __init__(self, model_class, num_folds=2, num_epochs=10, batch_size=32, num_classes=10, device='cuda' if torch.cuda.is_available() else 'cpu', loss_fn=None, optimizer_class=optim.Adam, optimizer_params={}):
        self.model_class = model_class  # model_class is a class that can be instantiated
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()  # Default to CrossEntropyLoss
        self.optimizer_class = optimizer_class  # Optimizer class, e.g., optim.Adam
        self.optimizer_params = optimizer_params  # Parameters for the optimizer

    def train(self, dataset):
        # Initialize model and optimizer
        model = self.model_class(self.num_classes).to(self.device)
        optimizer = self.optimizer_class(model.parameters(), **self.optimizer_params)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        for epoch in range(self.num_epochs):
            for _, (inputs, labels) in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            # Optional: print training progress
        self.model = model

    def validate(self, dataset):
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        logits_dict = {}
        with torch.no_grad():
            for idxs, (inputs, labels) in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.cpu()
                idxs = idxs.tolist()
                for idx, output in zip(idxs, outputs):
                    logits_dict[idx] = output
        return logits_dict

    def predict(self, dataset):
        self.dataset = IndexedDataset(dataset)  # Wrap the dataset to include indices
        kfold = KFold(n_splits=self.num_folds, shuffle=True)
        all_logits = [None] * len(self.dataset)  # To store logits for each sample

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f'Fold {fold+1}/{self.num_folds}')
            # Create data subsets
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            # Training
            self.train(self.dataset)

            # Validation
            logits_dict = self.validate(self.dataset)
            for idx, logit in logits_dict.items():
                all_logits[idx] = logit  # Store logit for each sample
            break

        # Concatenate logits and ensure the order matches the original dataset
        logits_list = [logit.unsqueeze(0) for logit in all_logits]
        logits = torch.cat(logits_list, dim=0)
        return logits


class ModelClass(nn.Module):
    def __init__(self, num_classes):
        super(ModelClass, self).__init__()
        self.model = PreActResNet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

def initialize_kfold_classifier(num_classes, device):
    return KFoldClassifier(
        model_class=ModelClass,
        num_folds=3,
        num_epochs=5,
        batch_size=256,
        num_classes=num_classes,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_class=Adam,
        optimizer_params={'lr': 3e-4, 'weight_decay': 5e-4},
    )



class CLIPClassifier:
    def __init__(
        self,
        model_name='ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        device='cuda',
        batch_size=64,
        label_dict=None,
        template="a photo of a {}"
    ):
        """
        Initializes the CLIPClassifier.

        Args:
            model_name (str): Name of the CLIP model to use.
            pretrained (str): Pretrained weights identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
            batch_size (int): Batch size for predictions.
            label_dict (dict): Dictionary mapping class indices to labels.
            template (str): Template string to format each class label.
        """
        self.device = device
        self.batch_size = batch_size
        self.label_dict = label_dict or {}
        self.template = template  # Store the template

        # Load the CLIP model and preprocessing transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )

        self.preprocess_tensor = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]
            ),  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
            *self.preprocess.transforms[:2],  # To match CLIP input scale assumptions
            *self.preprocess.transforms[4:]   # Skip convert PIL to tensor
        ])

        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Precompute text embeddings for labels using the template
        self.text_embeddings = self._compute_text_embeddings()

    def _compute_text_embeddings(self):
        """
        Computes text embeddings for all class labels using the provided template.

        Returns:
            Tensor: Normalized text embeddings of shape (num_classes, embedding_dim).
        """
        # Apply the template to each label
        formatted_texts = [self.template.format(label) for label in self.label_dict.values()]
        tokenized_texts = self.tokenizer(formatted_texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self, image):
        """
        Computes logits for a single image.

        Args:
            image (PIL.Image or Tensor): Input image.

        Returns:
            Tensor: Logits for each class.
        """
        # Preprocess and move image to the appropriate device
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = self.preprocess_tensor(image).to(self.device)
        else:
            raise TypeError("Image must be a PIL image or a torch Tensor.")

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute logits by taking dot product with text embeddings
            logits = image_features @ self.text_embeddings.T

        return logits.squeeze(0)  # Return logits for each class

    def predict(self, dataset):
        """
        Predict logits and return raw image features for images in the given dataset.

        Args:
            dataset (Dataset): PyTorch Dataset providing images (and optionally labels).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Logits tensor of shape (num_samples, num_classes).
                - Image features tensor of shape (num_samples, feature_dim).
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_logits = []
        all_image_features = []
        with torch.no_grad():
            for batch in dataloader:
                # Assuming batch = (images, labels) or just images
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch  # If dataloader returns only images

                images = images.to(self.device)
                # Preprocess images
                images = self.preprocess_tensor(images)
                # Encode images
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # Compute logits
                logits = image_features @ self.text_embeddings.T
                all_logits.append(logits.cpu())
                all_image_features.append(image_features.cpu())
        # Concatenate all logits and image features
        all_logits = torch.cat(all_logits, dim=0)
        all_image_features = torch.cat(all_image_features, dim=0)
        return all_logits, all_image_features


if __name__ == "__main__":
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define transforms to match CLIP's expected input size and normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(
        root='/data/datasets',
        train=False,
        transform=preprocess
    )

    label_dict = {value: key for key, value in test_dataset.class_to_idx.items()}

    # Instantiate the classifier with a custom template
    classifier = CLIPClassifier(
        model_name='ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        device=device,
        label_dict=label_dict,
        template="a photo of a {}"  # Example template
    )

    # Create a DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get a batch of data
    images, labels = next(iter(test_loader))
    images = images.to(device)

    # Pass the batch through the classifier
    with torch.no_grad():
        logits = classifier.forward(images)
        predictions = logits.argmax(dim=-1).cpu()

    # Map predictions and true labels to class names
    predicted_classes = [label_dict[pred.item()] for pred in predictions]
    true_classes = [label_dict[label.item()] for label in labels]

    # Print the predictions and true labels
    print("Predictions:", predicted_classes)
    print("True Labels:", true_classes)
