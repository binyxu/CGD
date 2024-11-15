import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import json
from collections import Counter

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import yaml
import logging
import time
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, all_acc, general_plot_for_epoch, given_dataloader_test
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import dataset_wrapper_with_transform

class NeuralDistillationDefense(defense):
    """
    Implement the knowledge distillation defense within BackdoorBench.
    """

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    @staticmethod
    def add_arguments(parser):
        # Basic arguments
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--interval', type=int, help='frequency of save model')

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int, help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/neural_distillation/config.yaml", help='the path of yaml')

        parser.add_argument('--temperature', type=float, help='temperature for distillation')
        parser.add_argument('--alpha', type=float, help='weight for the cross-entropy loss')
        parser.add_argument('--beta', type=float, help='weight for the repel loss')
        parser.add_argument('--sigma1', type=float, help='clean threshold')
        parser.add_argument('--sigma2', type=float, help='poison threshold')

        parser.add_argument('--tolerance', type=float, help='weight for the repel loss')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/cgd_{self.args.sigma1}_{self.args.sigma2}/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')
        # Set paths to logits.pt and clean_indices.json
        self.awc_path = os.path.join('record', result_file, 'AWC')
        self.logits_path = os.path.join(self.awc_path, 'logits.pt')
        self.clean_indices_path = os.path.join(self.awc_path, 'clean_indices.json')
        self.entropy_score = os.path.join(self.awc_path, 'entropy_score.json')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    def set_devices(self):
        self.device = self.args.device

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)
        result = self.result
        self.train_with_distillation(self.args, result)

        return result

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
    def compute_loss_subset(self, outputs, logits_clip_batch, labels, subset_idxs, criterion_kd, temperature, alpha, repel_factor, epsilon=1e-6):
        subset_outputs = outputs[subset_idxs]
        teacher_logits = logits_clip_batch[subset_idxs]
        teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = nn.functional.log_softmax(subset_outputs / temperature, dim=1)

        # Knowledge Distillation (KD) loss
        kd_loss = criterion_kd(student_log_probs, teacher_probs) * (temperature ** 2)

        original_labels = labels[subset_idxs]

        # 手动计算交叉熵损失并添加 epsilon
        student_probs = nn.functional.softmax(subset_outputs, dim=1)
        correct_class_probs = student_probs.gather(1, original_labels.unsqueeze(1)).squeeze(1)
        correct_class_probs = correct_class_probs + epsilon  # 添加 epsilon

        # 计算负的交叉熵损失
        repel_loss = torch.log(correct_class_probs).mean()
        return alpha * kd_loss + repel_factor * repel_loss

    def train_with_distillation(self, args, result):
        agg = Metric_Aggregator()
        # Load models
        logging.info('----------- Network Initialization --------------')
        student_model = generate_cls_model(args.model, args.num_classes)
        student_model.load_state_dict(self.result['model'])
        if "," in self.device:
            student_model = torch.nn.DataParallel(
                student_model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # e.g., "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{student_model.device_ids[0]}'
            student_model.to(self.args.device)
        else:
            student_model.to(self.args.device)
        logging.info('Finished model initialization...')

        # Initialize optimizer
        optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Define loss functions
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')

        # Load teacher logits and clean indices
        logging.info('----------- Loading Teacher Logits and Clean Indices --------------')
        data_to_save = torch.load(self.logits_path)
        logits_clip = data_to_save['logits_clip']  # Assuming logits_clip is the weak clean classifier's logits
        num_classes = data_to_save['num_classes']

        with open(self.clean_indices_path, 'r') as f:
            clean_indices = json.load(f)
            
        with open(self.entropy_score, 'r') as f:
            entropy_score = json.load(f)

        # Prepare the dataset
        logging.info('----------- Data Preparation --------------')
        tf_compose = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
        poisoned_data = result['bd_train']
        poisoned_data.wrap_img_transform = tf_compose

        distill_dataset = DistillationDataset(
            dataset=poisoned_data,
            logits_clip=logits_clip,
            clean_indices=clean_indices,
            entropy_score=entropy_score
        )

        # DataLoader
        batch_size = args.batch_size
        dataloader = torch.utils.data.DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        test_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        data_bd_testset = result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True, pin_memory=args.pin_memory)

        data_clean_testset = result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True, pin_memory=args.pin_memory)

        # Training Loop
        logging.info('----------- Starting Training with Distillation --------------')
        num_epochs = args.epochs
        temperature = args.temperature
        sigma1 = args.sigma1
        sigma2 = args.sigma2
        alpha = args.alpha
        beta = args.beta

        # Initialize variables for early stopping
        best_test_acc = -float('inf')
        prev_model_state_dict = copy.deepcopy(student_model.state_dict())
        best_epoch = -1

        train_loss_list = []
        train_mix_acc_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        
        def swap_largest_to_first(logits):
            # Get indices of the maximum values along axis 1
            max_indices = torch.argmax(logits, dim=1)
            
            # Gather the maximum values to swap them with the first element in each row
            max_values = logits[torch.arange(logits.size(0)), max_indices]
            first_values = logits[:, 0].clone()  # Clone to prevent in-place modification issues
            
            # Swap the max element with the element at position 0
            logits[:, 0] = max_values
            logits[torch.arange(logits.size(0)), max_indices] = first_values
            
            return logits

        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            student_model.train()
            running_loss = 0.0

            batch_loss_list = []
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []

            for batch_idx, batch in tqdm(enumerate(dataloader)):
                if batch_idx == 200:
                    break
                imgs = batch['img'].to(self.device)
                labels = batch['label'].to(self.device)
                idxs = batch['idx']
                is_clean = torch.tensor(batch['is_clean'], dtype=torch.bool).to(self.device)
                entropy_score = torch.stack(batch['entropy_score'], dim=-1).to(self.device)
                logits_clip_batch = batch['logits_clip'].to(self.device)
                poison_indicator = batch['poison_indicator'].to(self.device)
                original_targets = batch['original_targets'].to(self.device)

                optimizer.zero_grad()

                outputs = student_model(imgs)

                loss = 0.0

                # not_clean = torch.logical_or(torch.logical_and(entropy_score[:, 0] > 0.9, poison_indicator == 0), entropy_score[:, 1] < 0.1)
                # not_clean2 = torch.logical_or(torch.logical_and(entropy_score[:, 0] > 0.8, poison_indicator == 0), entropy_score[:, 1] < 0.2)
                # is_clean = ~not_clean2

                is_clean = torch.logical_and(entropy_score[:, 0] <= 1 - sigma1, entropy_score[:, 1] >= sigma1)
                if is_clean.any():
                    clean_idxs = is_clean.nonzero(as_tuple=True)[0]
                    clean_outputs = outputs[clean_idxs]
                    clean_labels_batch = labels[clean_idxs]
                    ce_loss = criterion_ce(clean_outputs, clean_labels_batch)
                    loss += ce_loss

                # For potentially poisoned samples, use soft labels from teacher logits and add repel loss
                # not_clean = torch.logical_or(torch.logical_and(entropy_score[:, 0] > 0.9, poison_indicator == 0), entropy_score[:, 1] < 0.1)
                not_clean = torch.logical_or(entropy_score[:, 0] > 1 - sigma2, entropy_score[:, 1] < sigma2)
                if not_clean.any():
                    not_clean_idxs = not_clean.nonzero(as_tuple=True)[0]
                    # deng_index = (poison_indicator == 1).nonzero(as_tuple=True)[0]
                    # logits_clip_batch[deng_index] = swap_largest_to_first(logits_clip_batch[deng_index])
                    loss += self.compute_loss_subset(
                        outputs, logits_clip_batch, labels, not_clean_idxs,
                        criterion_kd, temperature, alpha, beta
                    )
                
                # fault_label = torch.logical_or(entropy_score[:, 0] > 0.9, entropy_score[:, 1] < 0.1)
                # if fault_label.any():
                #     fault_label_idxs = fault_label.nonzero(as_tuple=True)[0]
                #     loss += self.compute_loss_subset(
                #         outputs, logits_clip_batch, labels, fault_label_idxs,
                #         criterion_kd, temperature, 0.0, 0.05
                #     )

                # clean_label_backdoor = entropy_score[:, 1] < 0.2
                # if clean_label_backdoor.any():
                #     clean_label_backdoor_idxs = clean_label_backdoor.nonzero(as_tuple=True)[0]
                #     loss += self.compute_loss_subset(
                #         outputs, logits_clip_batch, labels, clean_label_backdoor_idxs,
                #         criterion_kd, temperature, alpha, beta
                #     )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                batch_loss_list.append(loss.item())
                batch_predict_list.append(torch.max(outputs, -1)[1].detach().clone().cpu())
                batch_label_list.append(labels.detach().clone().cpu())
                batch_original_index_list.append(idxs.detach().clone().cpu())
                batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                batch_original_targets_list.append(original_targets.detach().clone().cpu())

            train_epoch_loss_avg_over_batch, \
            train_epoch_predict_list, \
            train_epoch_label_list, \
            train_epoch_poison_indicator_list, \
            train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                                torch.cat(batch_predict_list), \
                                                torch.cat(batch_label_list), \
                                                torch.cat(batch_poison_indicator_list), \
                                                torch.cat(batch_original_targets_list)

            train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

            train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
            train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
            train_clean_acc = all_acc(
                train_epoch_predict_list[train_clean_idx],
                train_epoch_label_list[train_clean_idx],
            )
            train_asr = all_acc(
                train_epoch_predict_list[train_bd_idx],
                train_epoch_label_list[train_bd_idx],
            )
            train_ra = all_acc(
                train_epoch_predict_list[train_bd_idx],
                train_epoch_original_targets_list[train_bd_idx],
            )

            # Evaluation
            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                student_model,
                data_clean_loader,
                data_bd_loader,
                args,
            )
            
            # Early Stopping Logic
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                # Do not save the model state here
            else:
                difference = (best_test_acc - test_acc) * 100  # Convert to percentage
                if difference >= args.tolerance:
                    logging.info(f"Early stopping at epoch {epoch + 1}, best test_acc {best_test_acc * 100:.2f}% at epoch {best_epoch + 1}")
                    # Restore model parameters to previous epoch
                    student_model.load_state_dict(prev_model_state_dict)
                    # Break the loop
                    break

            # Update prev_model_state_dict to current model state at the end of epoch
            prev_model_state_dict = copy.deepcopy(student_model.state_dict())

            agg({
                "epoch": epoch,

                "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                "train_acc": train_mix_acc,
                "train_acc_clean_only": train_clean_acc,
                "train_asr_bd_only": train_asr,
                "train_ra_bd_only": train_ra,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Train Acc": train_mix_acc_list,
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}distillation_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Train Loss": train_loss_list,
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}distillation_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}cgd_df.csv")

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                state_dict = {
                    "model": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_current": epoch,
                }
                torch.save(state_dict, args.checkpoint_save + "cgd_state_dict.pt")

        agg.summary().to_csv(f"{args.save_path}cgd_df_summary.csv")

        # Save the trained student model
        model_save_path = os.path.join(args.save_path, 'defense_result.pt')
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=student_model.cpu().state_dict(),
            save_path=args.save_path,
        )
        logging.info(f"Student model saved to {model_save_path}")

    def eval_step(
            self,
            netC,
            clean_test_dataloader,
            bd_test_dataloader,
            args,
    ):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra

from collections import Counter

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, logits_clip, clean_indices, entropy_score):
        self.dataset = dataset
        self.logits_clip = logits_clip
        self.entropy_score = entropy_score
        # Compute counts of clean indices (including duplicates)
        self.clean_indices_counter = Counter(clean_indices)

        # Build expanded indices list with oversampling
        self.expanded_indices = []

        dataset_length = len(self.dataset)
        all_indices = set(range(dataset_length))

        # Indices of potentially poisoned samples
        poisoned_indices = all_indices - set(self.clean_indices_counter.keys())

        # Add poisoned indices (each appears once)
        self.expanded_indices.extend(list(poisoned_indices))

        # Add clean indices (each appears as many times as they appear in clean_indices)
        for idx, count in self.clean_indices_counter.items():
            self.expanded_indices.extend([idx] * count)

        # Update clean indices set for checking in __getitem__
        self.clean_indices_set = set(self.clean_indices_counter.keys())

    def __len__(self):
        return len(self.expanded_indices)

    def __getitem__(self, idx):
        dataset_idx = self.expanded_indices[idx]
        img, label, original_index, poison_indicator, original_targets = self.dataset[dataset_idx]
        is_clean = dataset_idx in self.clean_indices_set

        data = {
            'img': img,
            'idx': dataset_idx,
            'is_clean': is_clean,
            'logits_clip': self.logits_clip[dataset_idx],
            'label': label,
            'poison_indicator': poison_indicator,
            'original_targets': original_targets,
            'entropy_score': self.entropy_score[f"{dataset_idx}"]
        }
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    NeuralDistillationDefense.add_arguments(parser)
    args = parser.parse_args()
    defense_method = NeuralDistillationDefense(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'your_attack_result_file_name'
    elif args.result_file is None:
        args.result_file = 'your_attack_result_file_name'
    result = defense_method.defense(args.result_file)
