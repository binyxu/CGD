import os
import torch
import numpy as np
import torchvision.transforms as transforms
from utils.data import load_dataset, PoisonedDataset, update_class2idx, DATASET_META, TEMPLATE_META, MAPPING_META
from utils.models import ModelClass, CLIPClassifier, KFoldClassifier
from utils.visualization import main_visualization, process_data, compute_logits


def main():
    device = "cuda:0"
    dataset_name = 'cifar10'
    resolution = 32
    pr = 0.05
    num_classes = DATASET_META[dataset_name]
    base_path = '/Your_Custom_BackdoorBench_Path/BackdoorBench/record'

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = load_dataset(name=dataset_name, root='/data/datasets', transform=transform)
    clean_labels = torch.tensor([y for _, y in dataset], dtype=torch.long)

    # Initialize CLIP classifier
    class_to_idx = update_class2idx(dataset.class_to_idx, MAPPING_META[dataset_name])
    label_dict = {value: key for key, value in class_to_idx.items()}
    clip_classifier = CLIPClassifier(
        model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device=device,
        label_dict=label_dict, batch_size=256, template=TEMPLATE_META[dataset_name]
    )

    for method in ["badnets"]:
        poisoned_data_dir = f'{base_path}/Your_Custom_Attack_Path/bd_train_dataset'
        out_dir = f'{poisoned_data_dir}/../AWC'

        poison_dataset = PoisonedDataset(dataset=dataset, poisoned_data_dir=poisoned_data_dir,
                                         transform=transform, name=method, pr=pr)
        poisoned_indices = torch.tensor(poison_dataset.poisoned_indices, dtype=torch.long)
        noise_indices = torch.tensor(poison_dataset.noise_indices, dtype=torch.long) if hasattr(poison_dataset, 'noise_indices') else None
        poison_labels = torch.tensor([y for _, y in poison_dataset], dtype=torch.long)

        logits_kfold_avg = compute_logits(poison_dataset, num_classes, device)
        logits_clip, all_image_features = clip_classifier.predict(poison_dataset)

        data_to_save = {
            'logits_kfold': logits_kfold_avg.detach().cpu(),
            'logits_clip': logits_clip.detach().cpu(),
            'poisoned_indices': poisoned_indices,
            'poison_labels': poison_labels,
            'clean_labels': clean_labels,
            'num_classes': num_classes,
            'image_features': all_image_features,
        }
        if noise_indices is not None:
            data_to_save['noise_indices'] = noise_indices

        os.makedirs(out_dir, exist_ok=True)
        torch.save(data_to_save, f'{out_dir}/logits.pt')
        print("Data successfully saved to 'logits.pt'")

        oversampled_indices = process_data(
            logits_kfold=logits_kfold_avg, logits_clip=logits_clip,
            poison_labels=poison_labels, clean_labels=clean_labels,
            poisoned_indices=poisoned_indices, noise_indices=noise_indices,
            out_dir=out_dir
        )

        print(f"Oversampled clean indices saved to {out_dir}/clean_indices.json")

if __name__ == "__main__":
    main()
