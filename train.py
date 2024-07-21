import polars as pl
from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from ranger import Ranger
import argparse
from accelerate import Accelerator
import time
import json
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
    return parser.parse_args()


def setup_environment(config: Any) -> None:
    np.random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'DETAIL'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('oofs', exist_ok=True)


def load_and_preprocess_data(config: Any) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
    data = pl.read_csv(f"{config.input_dir}/train_data.csv")
    data = drop_pk5090_duplicates(data)
    data = data.unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])

    SN = data['signal_to_noise'].to_numpy().astype('float32').reshape(-1, 2)
    SN = np.repeat(SN.min(-1), 2)
    dirty_data = data.filter(SN <= 1)
    data = data.filter(SN > 1)
    dirty_SN = dirty_data['signal_to_noise'].to_numpy().astype('float32').reshape(-1, 2)
    dirty_SN = np.repeat(dirty_SN.max(-1), 2)
    dirty_data = dirty_data.filter(dirty_SN > 1)

    label_names = [f"reactivity_{i:04d}" for i in range(206)]
    error_label_names = [f"reactivity_error_{i:04d}" for i in range(206)]

    data_dict = {
        'sequences': data.unique(subset=["sequence_id"], maintain_order=True)['sequence'].to_list(),
        'sequence_ids': data.unique(subset=["sequence_id"], maintain_order=True)['sequence_id'].to_list(),
        'labels': data[label_names].to_numpy().astype('float32').reshape(-1, 2, 206).transpose(0, 2, 1),
        'errors': data[error_label_names].to_numpy().astype('float32').reshape(-1, 2, 206).transpose(0, 2, 1),
        'SN': data['signal_to_noise'].to_numpy().astype('float32').reshape(-1, 2),
    }
    return data, dirty_data, data_dict


def prepare_data_loaders(config: Any, data_dict: Dict[str, Any], train_indices: np.ndarray, val_indices: np.ndarray
                         ) -> Tuple[DataLoader, DataLoader]:
    train_dataset = RNADataset(train_indices, data_dict, k=config.k, flip=config.use_flip_aug)
    val_dataset = RNADataset(val_indices, data_dict, train=False, k=config.k)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=Custom_Collate_Obj(), num_workers=min(config.batch_size, 16))
    val_loader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False,
                            collate_fn=Custom_Collate_Obj(), num_workers=min(config.batch_size, 16))
    return train_loader, val_loader


def configure_model_and_optimizer(config: Any, train_loader: DataLoader
                                  ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module, 
                                             torch.nn.Module, int, torch.optim.lr_scheduler._LRScheduler]:
    model = RibonanzaNet(config)
    optimizer = Ranger(model.parameters(), weight_decay=config.weight_decay, lr=config.learning_rate)
    criterion = torch.nn.L1Loss(reduction='none')
    val_criterion = torch.nn.L1Loss(reduction='none')
    cos_epoch = int(config.epochs * 0.75) - 1
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (config.epochs - cos_epoch) * len(train_loader) // config.gradient_accumulation_steps
    )
    return model, optimizer, criterion, val_criterion, cos_epoch, lr_schedule


def train_one_epoch(epoch: int, model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module, accelerator: Accelerator, cos_epoch: int, 
                    lr_schedule: torch.optim.lr_scheduler._LRScheduler, config: Any) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        src, masks, labels, SN, loss_masks = (
            batch['sequence'], batch['masks'].bool(), batch['labels'],
            batch['SN'].reshape(-1, 1, 2) >= 1, batch['loss_masks']
        )
        loss_masks *= SN
        with accelerator.autocast():
            output = model(src, masks)
            loss = criterion(output, labels)[loss_masks].mean()
        accelerator.backward(loss / config.gradient_accumulation_steps)
        if (batch['index'] + 1) % config.gradient_accumulation_steps == 0:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            if epoch > cos_epoch:
                lr_schedule.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_one_epoch(epoch: int, model: torch.nn.Module, val_loader: DataLoader, val_criterion: torch.nn.Module, 
                       accelerator: Accelerator, config: Any) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    val_loss = 0
    preds, gts, val_loss_masks = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val"):
            src, masks, labels, loss_masks = (
                batch['sequence'], batch['masks'].bool(), batch['labels'], batch['loss_masks']
            )
            src_flipped = src.clone()
            for batch_idx in range(len(src)):
                src_flipped[batch_idx, :batch['length'][batch_idx]] = src_flipped[batch_idx, :batch['length'][batch_idx]].flip(0)
            with accelerator.autocast():
                output = model(src, masks)
                if config.use_flip_aug:
                    flipped_output = model(src_flipped, masks)
                    for batch_idx in range(len(flipped_output)):
                        flipped_output[batch_idx, :batch['length'][batch_idx]] = flipped_output[batch_idx, :batch['length'][batch_idx]].flip(0)
                    output = (flipped_output + output) / 2
            loss = val_criterion(output, labels)[loss_masks].mean()
            all_output = accelerator.gather(F.pad(output, (0, 0, 0, 206 - src.shape[1]), value=0))
            all_labels = accelerator.gather(F.pad(labels, (0, 0, 0, 206 - src.shape[1]), value=0))
            all_masks = accelerator.gather(F.pad(loss_masks, (0, 0, 0, 206 - src.shape[1]), value=0))
            preds.append(all_output)
            gts.append(all_labels)
            val_loss_masks.append(all_masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss, torch.cat(preds), torch.cat(gts), torch.cat(val_loss_masks)


def save_model_and_metrics(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                           train_loss: float, val_loss: float, preds: torch.Tensor, gts: torch.Tensor, 
                           val_loss_masks: torch.Tensor, best_val_loss: float, config: Any, logger: CSVLogger, 
                           accelerator: Accelerator) -> float:
    if accelerator.is_local_main_process:
        val_loss = val_criterion(
            preds[val_loss_masks], 
            gts[val_loss_masks]
        ).mean().item()
        logger.log([epoch, train_loss, val_loss])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(accelerator.unwrap_model(model).state_dict(), f"models/model{config.fold}.pt")
            with open(f"oofs/{config.fold}.pkl", "wb+") as file:
                pickle.dump({
                    "preds": preds.cpu().numpy(),
                    "gts": gts.cpu().numpy(),
                    "val_loss_masks": val_loss_masks.cpu().numpy()
                }, file)
    return best_val_loss


def main():
    start_time = time.time()
    args = parse_args()
    config = load_config_from_yaml(args.config_path)
    accelerator = Accelerator(mixed_precision='fp16', device_placement=True, distributed_type='MULTI_GPU')
    setup_environment(config)

    data, dirty_data, data_dict = load_and_preprocess_data(config)

    kfold = StratifiedKFold(n_splits=config.nfolds, shuffle=True, random_state=0)
    fold_indices = {
        i: (train_index, test_index) 
        for i, (train_index, test_index) in enumerate(
            kfold.split(np.arange(len(data) // 2), data['dataset_name'].to_list())
        )
    }
    train_indices, val_indices = fold_indices[config.fold]

    if config.use_data_percentage < 1:
        size = int(config.use_data_percentage * len(train_indices))
        train_indices = np.random.choice(train_indices, size, replace=False)

    if config.use_dirty_data:
        data_dict['sequences'] += dirty_data.unique(subset=["sequence_id"], maintain_order=True)['sequence'].to_list()
        data_dict['sequence_ids'] += dirty_data.unique(subset=["sequence_id"], maintain_order=True)['sequence_id'].to_list()
        data_dict['labels'] = np.concatenate([
            data_dict['labels'], 
            dirty_data[label_names].to_numpy().astype('float32').reshape(-1, 2, 206).transpose(0, 2, 1)
        ])
        data_dict['errors'] = np.concatenate([
            data_dict['errors'], 
            dirty_data[error_label_names].to_numpy().astype('float32').reshape(-1, 2, 206).transpose(0, 2, 1)
        ])
        data_dict['SN'] = np.concatenate([
            data_dict['SN'], 
            dirty_data['signal_to_noise'].to_numpy().astype('float32').reshape(-1, 2)
        ])
        train_indices = np.concatenate([train_indices, np.arange(len(data) // 2, len(data) // 2 + len(dirty_data) // 2)])

    train_loader, val_loader = prepare_data_loaders(config, data_dict, train_indices, val_indices)
    model, optimizer, criterion, val_criterion, cos_epoch, lr_schedule = configure_model_and_optimizer(config, train_loader)

    model, optimizer, train_loader, val_loader, lr_schedule = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_schedule
    )

    best_val_loss = np.inf
    logger = CSVLogger(['epoch', 'train_loss', 'val_loss'], f'logs/fold{config.fold}.csv')

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, accelerator, cos_epoch, lr_schedule, config)
        val_loss, preds, gts, val_loss_masks = validate_one_epoch(epoch, model, val_loader, val_criterion, accelerator, config)
        best_val_loss = save_model_and_metrics(epoch, model, optimizer, train_loss, val_loss, preds, gts, val_loss_masks, best_val_loss, config, logger, accelerator)

    if accelerator.is_local_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), f"models/model{config.fold}_lastepoch.pt")
        elapsed_time = time.time() - start_time
        with open("run_stats.json", 'w') as file:
            json.dump({'Total_execution_time': elapsed_time}, file, indent=4)


if __name__ == "__main__":
    main()
