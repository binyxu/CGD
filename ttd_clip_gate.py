#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, time, logging, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import open_clip
import pandas as pd

sys.path.append(os.getcwd())
# ===== BackdoorBench imports =====
from defense.base import defense
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, general_plot_for_epoch
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.prompt_template import dataset_prompts

import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque

class OnlineTTGate:
    def __init__(self, strong_model, clip_model, text_features, preprocess_for_clip,
                 num_classes, device="cuda",
                 tau_global=0.30, k_std=0.5, guard_min_share=1e-3,
                 da_alpha=0.01, da_beta=1.0,
                 use_lin_head=False, feat_dim=None, lr=5e-3, weight_decay=0.0,
                 consensus_conf=0.7,
                 ema_alpha_conf=0.05,
                 use_online_ema=True):
        self.strong = strong_model.eval()
        self.clip = clip_model.eval()
        self.text_features = F.normalize(text_features, dim=-1).to(device)   # [C, D]
        self.preprocess = preprocess_for_clip
        self.num_classes = int(num_classes)
        self.device = device

        # ----- gating params -----
        self.tau_global = float(tau_global)
        self.k_std = float(k_std)
        self.guard_min_share = float(guard_min_share)
        self.ema_alpha_conf = float(ema_alpha_conf)
        self.use_online_ema = bool(use_online_ema)

        # ----- DA (class prior) -----
        self.da_alpha = float(da_alpha)
        self.da_beta  = float(da_beta)
        self.prior_ema = torch.full((self.num_classes,), 1.0/self.num_classes, device=device)

        # ----- per-class conf stats -----
        self.conf_mean = torch.zeros(self.num_classes, device=device)
        self.conf_m2   = torch.zeros(self.num_classes, device=device)
        self.conf_cnt  = torch.zeros(self.num_classes, device=device)

        # ----- online linear head (optional) -----
        self.use_lin_head = bool(use_lin_head)
        if self.use_lin_head:
            D = feat_dim if feat_dim is not None else self.text_features.size(1)
            self.W = nn.Parameter(torch.zeros(D, self.num_classes, device=device, dtype=torch.float32))
            self.opt = torch.optim.SGD([self.W], lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.consensus_conf = float(consensus_conf)
        else:
            self.W = None
            self.opt = None

    @torch.no_grad()
    def _encode_image(self, x):
        x_clip = self.preprocess(x)                      # 张量版 CLIP 预处理
        z = self.clip.encode_image(x_clip)               # [B, D]
        return F.normalize(z, dim=-1)

    def _clip_logits(self, z):
        if self.use_lin_head and self.W is not None and self.W.requires_grad is False:
            # 若外部把 W 冻结了，这里兼容
            pass
        # text head
        Lc_text = (z @ self.text_features.T) * self.clip.logit_scale.exp()
        if self.use_lin_head and self.W is not None:
            # 线性头（在线学得）与文本头相加/拼加均可；这里选择相加，保持维度一致
            Lc_lin = z @ self.W
            return Lc_text + Lc_lin
        return Lc_text

    def _da_bias(self):
        # b_c = -log(prior_c) * beta
        return - self.da_beta * torch.log(self.prior_ema.clamp_min(1e-8))

    def _update_prior_ema(self, p):
        # p: [B, C] softmax
        with torch.no_grad():
            batch_mean = p.mean(dim=0)
            self.prior_ema = (1 - self.da_alpha) * self.prior_ema + self.da_alpha * batch_mean

    def _update_conf_stats(self, s_idx, conf):
        # s_idx: [B], conf: [B] （CLIP 在 strong-top1 上的概率）
        with torch.no_grad():
            for c in s_idx.unique():
                m = (s_idx == c)
                if m.any():
                    v = conf[m].mean()
                    # EMA of mean & second moment
                    self.conf_cnt[c] = (1 - self.ema_alpha_conf) * self.conf_cnt[c] + self.ema_alpha_conf * 1.0
                    self.conf_mean[c] = (1 - self.ema_alpha_conf) * self.conf_mean[c] + self.ema_alpha_conf * v
                    self.conf_m2[c]   = (1 - self.ema_alpha_conf) * self.conf_m2[c]   + self.ema_alpha_conf * (v * v)

    def _per_class_tau(self, s_idx):
        # 若禁用在线 EMA，则直接使用全局 τ
        if not self.use_online_ema:
            return torch.full((s_idx.size(0),), self.tau_global,
                            device=self.device, dtype=torch.float32)

        # τ_c = min(τ_global, mean_c - k*std_c)；若 prior 很低 -> τ_c = 0
        var = (self.conf_m2 - self.conf_mean * self.conf_mean).clamp_min(0.0)
        std = var.sqrt()
        
        if args.num_classes > 200:
            base = (self.conf_mean - self.k_std * std).clamp(0, 1)
            tau_c = (base + self.tau_global).clamp(0, 1)
        else:
            tau_c = (self.conf_mean - self.k_std * std).clamp_min(0.0)
            tau_c = torch.minimum(tau_c, torch.full_like(tau_c, self.tau_global))
            tau_c = torch.where(self.prior_ema < self.guard_min_share,
                                torch.zeros_like(tau_c), tau_c)
        return tau_c[s_idx]   # [B]

    def _maybe_update_linear_head(self, z, y_s, p_c):
        if not self.use_lin_head: 
            return
        with torch.no_grad():
            y_c = p_c.argmax(dim=1)
            conf = p_c.max(dim=1).values
            mask = (y_s == y_c) & (conf >= self.consensus_conf)
        if not mask.any(): 
            return

        with torch.enable_grad():
            z_sup = z[mask].detach()   # 特征视作常量
            y_sup = y_s[mask].detach()
            logits = z_sup @ self.W    # W 是 nn.Parameter
            loss = F.cross_entropy(logits, y_sup)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()


    @torch.no_grad()
    def step(self, x):
        """
        x: [B,C,H,W]。返回 gating 后的最终 logits（在 self.device 上）。
        """
        x = x.to(self.device, non_blocking=True)
        amp = torch.cuda.is_available()

        # 1) strong
        with torch.cuda.amp.autocast(enabled=amp):
            S = self.strong(x)                            # [B,C]
        S = S.float()

        # 2) CLIP
        z = self._encode_image(x).float()                # [B,D]
        Lc = self._clip_logits(z).float()                # [B,C]

        # 3) 在线 DA（可关）
        p_clip = F.softmax(Lc, dim=1)
        if self.use_online_ema:
            self._update_prior_ema(p_clip.float())
            Lc = Lc + self._da_bias().to(Lc.dtype)

        # 4) gating
        s_idx = S.argmax(dim=1)                          # [B]
        p_da = F.softmax(Lc, dim=1)
        conf = p_da[torch.arange(x.size(0), device=self.device), s_idx]
        tau_eff = self._per_class_tau(s_idx)

        out = S.clone()
        if out.dtype != Lc.dtype:
            Lc = Lc.to(out.dtype)
        mask = conf < tau_eff
        out[mask] = Lc[mask]

        # 5) 在线统计与（可选）线性头
        if self.use_online_ema:
            self._update_conf_stats(s_idx, conf)
        self._maybe_update_linear_head(z, s_idx, p_da)

        return out



class TTGatedModel(nn.Module):
    """
    Gate-by-CLIP at test time:
      - If CLIP prob of strong top-1 class < tau -> use CLIP logits
      - Else use strong logits
    """
    def __init__(self, strong_model: nn.Module,
                 clip_model, text_features: torch.Tensor,
                 preprocess_for_clip, tau: float):
        super().__init__()
        self.strong = strong_model
        self.clip = clip_model
        self.register_buffer("text_features", F.normalize(text_features, dim=-1))
        self.preprocess_for_clip = preprocess_for_clip
        self.tau = float(tau)

    @torch.no_grad()
    def _clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        x_clip = self.preprocess_for_clip(x)                  # tensor-based CLIP preprocess
        img_feat = self.clip.encode_image(x_clip)             # [B, D]
        img_feat = F.normalize(img_feat, dim=-1)
        logits = (img_feat @ self.text_features.T) * self.clip.logit_scale.exp()
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.strong(x)                                    # strong logits
        with torch.no_grad():
            Lc = self._clip_logits(x)                         # CLIP logits
            Pc = F.softmax(Lc, dim=1)
            s_idx = S.argmax(dim=1)                           # strong top-1
            batch = torch.arange(x.size(0), device=x.device)
            conf_strong_under_clip = Pc[batch, s_idx]         # CLIP prob of strong top-1
            mask = (conf_strong_under_clip < self.tau)
        # combine logits per-sample
        out = S.clone()
        out[mask] = Lc[mask]
        return out

class ttd_clip_gate(defense):
    """
    Test-time backdoor defense via CLIP gating (weak teacher).
    - Sweep tau and report CA & ASR per tau.
    """
    @staticmethod
    def _compute_counts_and_confusion(pred: torch.Tensor, y: torch.Tensor, num_classes: int):
        """
        返回：
        counts_gt[c]     = 该类在GT中的样本数
        counts_pred[c]   = 该类在CLIP预测中的样本数
        confusion[g, p]  = GT=g 且 预测=p 的样本数（行=GT，列=CLIP_PRED）
        """
        y = y.view(-1).cpu().to(torch.long)
        pred = pred.view(-1).cpu().to(torch.long)

        counts_gt = torch.bincount(y, minlength=num_classes)
        counts_pred = torch.bincount(pred, minlength=num_classes)

        idx = y * num_classes + pred
        confusion = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        return counts_gt, counts_pred, confusion

    def _dump_clip_bias(self, Lc: torch.Tensor, y: torch.Tensor, split_tag: str):
        """
        在 self.args.save_path 下保存：
        - clip_bias_{split_tag}.csv          ：每类 GT/Pred 计数与占比、ratio、per-class acc
        - clip_confusion_{split_tag}.csv     ：混淆矩阵（行=GT，列=CLIP_PRED）
        """
        num_classes = int(self.args.num_classes)
        save_path = self.args.save_path

        with torch.no_grad():
            pred = Lc.argmax(dim=1)
        counts_gt, counts_pred, confusion = self._compute_counts_and_confusion(pred, y, num_classes)

        total = counts_gt.sum().item()
        eps = 1e-8
        diag = confusion.diag()
        per_class_acc = (diag.float() / torch.clamp(counts_gt.float(), min=1)).numpy()

        df = pd.DataFrame({
            "class_id": np.arange(num_classes, dtype=int),
            "gt_count": counts_gt.numpy().astype(int),
            "clip_pred_count": counts_pred.numpy().astype(int),
        })
        df["gt_share"] = df["gt_count"] / max(1, total)
        df["pred_share"] = df["clip_pred_count"] / max(1, total)
        df["pred_gt_ratio"] = (df["clip_pred_count"] + eps) / (df["gt_count"] + eps)
        df["per_class_acc"] = per_class_acc

        # 保存明细
        csv_bias = os.path.join(save_path, f"clip_bias_{split_tag}.csv")
        df.to_csv(csv_bias, index=False)

        # 混淆矩阵保存（行=GT, 列=CLIP_PRED）
        csv_conf = os.path.join(save_path, f"clip_confusion_{split_tag}.csv")
        conf_df = pd.DataFrame(confusion.numpy())
        conf_df.index.name = "GT"
        conf_df.columns = [f"pred_{i}" for i in range(num_classes)]
        conf_df.to_csv(csv_conf)

        # 日志里给个简要摘要
        over_idx = int(df["pred_gt_ratio"].idxmax())
        under_idx = int(df["pred_gt_ratio"].idxmin())
        logging.info(f"[CLIP bias/{split_tag}] total={total} | overall_acc={float((pred.cpu()==y.cpu()).float().mean()):.4f}")
        logging.info(f"[CLIP bias/{split_tag}] most over-pred class={over_idx} ratio={df.loc[over_idx,'pred_gt_ratio']:.3f} "
                     f"(gt={df.loc[over_idx,'gt_count']}, pred={df.loc[over_idx,'clip_pred_count']})")
        logging.info(f"[CLIP bias/{split_tag}] most under-pred class={under_idx} ratio={df.loc[under_idx,'pred_gt_ratio']:.3f} "
                     f"(gt={df.loc[under_idx,'gt_count']}, pred={df.loc[under_idx,'clip_pred_count']})")
        logging.info(f"[CLIP bias/{split_tag}] saved bias csv -> {csv_bias}")
        logging.info(f"[CLIP bias/{split_tag}] saved confusion csv -> {csv_conf}")
    
    @staticmethod
    def add_arguments(parser):
        # basics (BackdoorBench style)
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("--pin_memory", type=lambda x: str(x).lower() in ['true','1'], default=True)
        parser.add_argument("--non_blocking", type=lambda x: str(x).lower() in ['true','1'], default=True)
        parser.add_argument('--amp', type=lambda x: str(x).lower() in ['true','1'], default=True)

        parser.add_argument('--checkpoint_load', type=str)
        parser.add_argument('--checkpoint_save', type=str)
        parser.add_argument('--log', type=str)
        parser.add_argument("--dataset_path", type=str)
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--result_file', type=str)

        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument('--model', type=str)

        parser.add_argument('--frequency_save', type=int, default=0)
        parser.add_argument('--random_seed', type=int, default=0)
        parser.add_argument('--yaml_path', type=str, default="./config/defense/ft-sam/config.yaml")

        # TT gating hyper-params
        parser.add_argument('--tau', type=float, default=0.30, help='single threshold (also used for fixed result)')
        parser.add_argument('--tau_min', type=float, default=0.0)
        parser.add_argument('--tau_max', type=float, default=0.9)
        parser.add_argument('--tau_num', type=int, default=10)

        # CLIP options
        parser.add_argument('--clip_arch', type=str, default='ViT-B-32')
        parser.add_argument('--clip_ckpt', type=str, default='laion2b_s34b_b79K')
        parser.add_argument(
            '--online_ema',
            type=lambda x: str(x).lower() in ['true','1'],
            default=True,
            help='是否启用在线 EMA（DA-EMA 与按类置信统计）。默认开启。'
        )

    def __init__(self, args):
        import yaml
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults
        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args
        if 'result_file' in args.__dict__ and args.result_file is not None:
            self.set_result(args.result_file)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        ema_str = "_ema" if self.args.online_ema else "_no_ema"
        save_path = 'record/' + result_file + f'/defense/ttd_clip_gate{ema_str}/'
        os.makedirs(save_path, exist_ok=True)
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = os.path.join(save_path, 'checkpoint/')
            os.makedirs(self.args.checkpoint_save, exist_ok=True)
        if self.args.log is None:
            self.args.log = os.path.join(save_path, 'log/')
            os.makedirs(self.args.log, exist_ok=True)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_devices(self):
        self.device = self.args.device

    def set_logger(self):
        from pprint import pformat
        from utils.log_assist import get_git_info
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        fileHandler = logging.FileHandler(self.args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter); logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(); consoleHandler.setFormatter(logFormatter); logger.addHandler(consoleHandler)
        logger.setLevel(logging.INFO)
        logging.info(pformat(self.args.__dict__))
        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    @torch.no_grad()
    def _eval_loader(self, model: nn.Module, loader: DataLoader, criterion, device: str, non_blocking=True):
        """Return (loss_avg, acc)."""
        model.eval()
        metrics = {'test_correct': 0, 'test_total': 0, 'test_loss_sum_over_batch': 0.0}
        for x, y, *_ in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            logits = model(x)
            loss = criterion(logits, y.long())
            pred = logits.argmax(dim=1)
            metrics['test_correct'] += pred.eq(y).sum().item()
            metrics['test_total'] += y.size(0)
            metrics['test_loss_sum_over_batch'] += float(loss.item())
        loss_avg = metrics['test_loss_sum_over_batch'] / max(1, len(loader))
        acc = metrics['test_correct'] / max(1, metrics['test_total'])
        return loss_avg, acc

    def _build_clip(self, test_tran):
        # 1) CLIP backbone
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            self.args.clip_arch, pretrained=self.args.clip_ckpt
        )
        clip_model.to(self.device); clip_model.eval()

        # 2) CLIP text features from dataset_prompts
        ds_name = self.args.dataset.lower()
        prompts = {k.lower(): v for k, v in dataset_prompts.items()}
        if ds_name not in prompts:
            raise ValueError(f"No prompts for dataset={self.args.dataset}")
        classes = prompts[ds_name]['classes']
        templates = prompts[ds_name]['templates']
        texts = [tmpl.format(c) for c in classes for tmpl in templates]
        # tokenize & encode in small batches
        tokens = open_clip.tokenize(texts).to(self.device)
        feats = []
        bs = 128
        with torch.no_grad():
            for i in range(0, tokens.size(0), bs):
                feats.append(clip_model.encode_text(tokens[i:i+bs]))
        text_features = torch.cat(feats, dim=0).reshape(len(classes), len(templates), -1).mean(dim=1)
        text_features = F.normalize(text_features, dim=-1)

        # 3) Tensor-based CLIP preprocess that cancels dataset Normalize first (your recipe)
        #    Normalize(mean=-mean/std, std=1/std) -> Resize/CenterCrop -> (skip ToTensor) -> CLIP Normalize
        norm = test_tran.transforms[-1]  # assume last is transforms.Normalize
        mean = torch.tensor(norm.mean)
        std  = torch.tensor(norm.std)
        clip_preprocess_tensor = transforms.Compose([
            transforms.Normalize(mean=(-mean / std), std=(1.0 / std)),
            *clip_preprocess.transforms[:2],   # Resize & CenterCrop to CLIP's size
            *clip_preprocess.transforms[4:]    # Skip PIL->Tensor
        ])
        return clip_model, text_features, clip_preprocess_tensor

    @torch.no_grad()
    def _precompute_logits(self, strong, clip_model, text_features, preprocess_for_clip, loader):
        """
        对给定 loader 一次性计算并缓存：
          - S: strong 模型的 logits [N, C]
          - Lc: CLIP 的 logits (image-text) [N, C]
          - y: 标签 [N]
        全部搬到 CPU，后续 sweep τ 时直接张量计算。
        """
        device = self.args.device
        amp = bool(self.args.amp)
        all_S, all_Lc, all_y = [], [], []

        strong.eval(); clip_model.eval()
        for x, y, *_ in loader:
            x = x.to(device, non_blocking=self.args.non_blocking)

            # strong logits
            with torch.cuda.amp.autocast(enabled=amp):
                S = strong(x)                                   # [B, C]

            # CLIP logits
            x_clip = preprocess_for_clip(x)                     # 仍是张量流程
            img_feat = clip_model.encode_image(x_clip)          # [B, D]
            img_feat = F.normalize(img_feat, dim=-1)
            Lc = (img_feat @ text_features.T) * clip_model.logit_scale.exp()

            all_S.append(S.detach().float().cpu())
            all_Lc.append(Lc.detach().float().cpu())
            all_y.append(y.cpu())

        S = torch.cat(all_S, dim=0)
        Lc = torch.cat(all_Lc, dim=0)
        y = torch.cat(all_y, dim=0)
        return S, Lc, y

    @staticmethod
    def _gate_logits_from_cache(S: torch.Tensor, Lc: torch.Tensor, tau: float) -> torch.Tensor:
        """
        给定缓存的 strong/CLIP logits 与阈值 tau，返回 gating 后的最终 logits（vectorized）。
        规则：若 CLIP 对 strong-top1 的概率 < tau，则用 CLIP logits，否则用 strong logits。
        """
        # CLIP 概率
        Pc = F.softmax(Lc, dim=1)                               # [N, C]
        s_idx = S.argmax(dim=1)                                 # [N]
        conf = Pc.gather(1, s_idx.view(-1, 1)).squeeze(1)       # [N]
        mask = conf.lt(tau)                                     # [N]
        out = S.clone()
        out[mask] = Lc[mask]
        return out

    @staticmethod
    def _acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
        pred = logits.argmax(dim=1).cpu()
        return float((pred == labels.cpu()).float().mean().item())

    def mitigation(self):
        args = self.args
        self.set_devices()
        fix_random(args.random_seed)

        # === (1) load strong teacher ===
        strong = generate_cls_model(args.model, args.num_classes)
        strong.load_state_dict(self.result['model'])
        strong.to(args.device).eval()

        # === (2) dataloaders (clean/bd) ===
        criterion = argparser_criterion(args)  # 仅为接口对齐，这里不直接用 loss
        test_tran = get_transform(args.dataset, *(args.input_height, args.input_width), train=False)
        data_bd = self.result['bd_test'];    data_bd.wrap_img_transform = test_tran
        data_cl = self.result['clean_test']; data_cl.wrap_img_transform = test_tran
        ld_bd = DataLoader(data_bd, batch_size=args.batch_size, num_workers=args.num_workers,
                        drop_last=False, shuffle=True, pin_memory=args.pin_memory)
        ld_cl = DataLoader(data_cl, batch_size=args.batch_size, num_workers=args.num_workers,
                        drop_last=False, shuffle=True, pin_memory=args.pin_memory)

        # === (3) CLIP weak teacher + text features + preprocess ===
        clip_model, text_features, preprocess_for_clip = self._build_clip(test_tran)

        # === (4) 在线门控：逐阈值评测（每个 τ 初始化一个全新的 OnlineTTGate 状态） ===
        taus = np.linspace(args.tau_min, args.tau_max, int(args.tau_num))
        agg_all = Metric_Aggregator()
        CA_list, ASR_list = [], []

        # 在线自适应的默认超参（如未在 args 中声明，则使用这里的默认值）
        k_std            = getattr(args, "k_std", 0.3)
        guard_min_share  = getattr(args, "guard_min_share", 1e-3)
        da_alpha         = getattr(args, "da_alpha", 0.01)
        da_beta          = getattr(args, "da_beta", 0.5)
        use_lin_head     = getattr(args, "online_lin_head", False)
        lin_lr           = getattr(args, "lin_lr", 1e-3)
        lin_wd           = getattr(args, "lin_wd", 0.0)
        consensus_conf   = getattr(args, "consensus_conf", 0.5)
        ema_alpha_conf   = getattr(args, "ema_alpha_conf", 0.05)
        online_ema = getattr(args, "online_ema", True)

        for tau in taus:
            tau = float(tau)

            # —— 为当前 τ 初始化一个在线门控器（状态独立） ——
            gate = OnlineTTGate(
                strong_model=strong, clip_model=clip_model, text_features=text_features,
                preprocess_for_clip=preprocess_for_clip, num_classes=args.num_classes,
                device=args.device,
                tau_global=tau, k_std=k_std, guard_min_share=guard_min_share,
                da_alpha=da_alpha, da_beta=da_beta,
                use_lin_head=use_lin_head, feat_dim=text_features.size(1),
                lr=lin_lr, weight_decay=lin_wd,
                consensus_conf=consensus_conf, ema_alpha_conf=ema_alpha_conf,
                use_online_ema=online_ema
            )

            # === (4.a) 在线评测 Clean（流式，边推理边更新状态） ===
            correct_cl, total_cl = 0, 0
            for x, y, *_ in ld_cl:
                x = x.to(args.device, non_blocking=args.non_blocking)
                y = y.to(args.device, non_blocking=args.non_blocking)
                # 注意：不要用 torch.no_grad()，以便在线线性头可做一次小步 SGD
                logits = gate.step(x)                   # 已完成门控 + 在线自适应更新
                pred = logits.argmax(dim=1)
                correct_cl += pred.eq(y).sum().item()
                total_cl   += y.size(0)
            ca = correct_cl / max(1, total_cl)

            # === (4.b) 在线评测 Backdoor（继续沿用同一 gate 状态） ===
            correct_bd, total_bd = 0, 0
            for x, y, *_ in ld_bd:
                x = x.to(args.device, non_blocking=args.non_blocking)
                y = y.to(args.device, non_blocking=args.non_blocking)  # 在 BB 中这里是目标标签
                logits = gate.step(x)
                pred = logits.argmax(dim=1)
                correct_bd += pred.eq(y).sum().item()
                total_bd   += y.size(0)
            asr = correct_bd / max(1, total_bd)  # y 为目标标签 -> acc 即 ASR

            CA_list.append(float(ca)); ASR_list.append(float(asr))
            agg_all({"tau": float(tau), "CA": float(ca), "ASR": float(asr)})
            agg_all.to_dataframe().to_csv(f"{args.save_path}/tau_sweep_df.csv", index=False)

            general_plot_for_epoch(
                {"Clean Acc (CA)": CA_list, "ASR": ASR_list},
                save_path=f"{args.save_path}/tau_sweep_acc_asr.png",
                ylabel="percentage",
            )

        # === (5) 保存 strong 权重（wrapper/在线状态仅测试期使用，不落盘） ===
        save_defense_result(
            model_name=args.model, num_classes=args.num_classes,
            model=strong.cpu().state_dict(),
            save_path=args.save_path,
        )
        return {"model": strong}


    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        return self.mitigation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=sys.argv[0])
    ttd_clip_gate.add_arguments(parser)
    args = parser.parse_args()
    method = ttd_clip_gate(args)
    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'
    method.defense(args.result_file)
