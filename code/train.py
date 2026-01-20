import os
import random
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import CAMDTA
from config import HyperParameter
from dataset import CustomDataSet, batch2tensor, my_collate_fn

from sklearn.metrics import r2_score
from tqdm import tqdm
from math import sqrt
from scipy import stats
import csv
import wandb


def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return sum_m / pair
    else:
        return 0

    
def regression_scores(label, pred, is_valid=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred)**2).mean(axis=0)
    rmse = sqrt(mse)
    if is_valid:
        ci = -1
    else:
        ci = cindex_score(label, pred)
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(mse, 6), round(rmse, 6), round(ci, 6), round(r2, 6), round(pearson, 6), round(spearman, 6)


def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)

    
def test(model, dataloader, is_valid=True):
    model.eval()
    preds = []
    labels = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
        with torch.no_grad():
            if hasattr(model, 'module'):
                pred = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False)
            else:
                pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(labels, preds, is_valid)
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CAMDTA model for a specific fold')
    parser.add_argument('--fold', type=int, required=True, 
                        help='Fold index to train (0-4 for 5-fold CV)')
    parser.add_argument('--cuda', type=str, default=None,
                        help='CUDA device ID (e.g., "0", "1"). Overrides config.py setting')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to override config.py setting')
    parser.add_argument('--running_set', type=str, default=None,
                        help='Running set to override config.py setting')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config.py setting)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size to use (overrides config.py setting)')
    parser.add_argument('--wandb_project', type=str, default='CAMDTA',
                        help='Weights & Biases project name (default: CAMDTA)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username (optional)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--use_esmc', type=lambda x: x.lower() == 'true', default=None,
                        help='Use ESM-C (True) or ESM2 (False). Overrides config.py setting')
    parser.add_argument('--esmc_model', type=str, default=None, choices=['esmc_300m', 'esmc_600m', 'esmc_6b'],
                        help='ESM-C model variant (esmc_300m, esmc_600m, esmc_6b). Overrides config.py setting')
    parser.add_argument('--num_experts', type=int, default=None,
                        help='Number of experts in MoE (default: 4)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of experts to select per sample (default: 2)')
    parser.add_argument('--moe_noise_std', type=float, default=None,
                        help='Noise std for MoE exploration (default: 0.1, 0 to disable)')
    parser.add_argument('--load_balance_weight', type=float, default=None,
                        help='Weight for load balancing loss (default: 0.01, 0 to disable)')
    args = parser.parse_args()
    
    fold_i = args.fold

    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    
    if args.use_esmc is not None:
        hp.use_esmc = args.use_esmc
        if not hp.use_esmc:
            hp.protvec_dim = 1280
    
    if args.esmc_model is not None:
        hp.esmc_model = args.esmc_model
        if hp.esmc_model == "esmc_300m":
            hp.protvec_dim = 960
        elif hp.esmc_model == "esmc_600m":
            hp.protvec_dim = 1152
        elif hp.esmc_model == "esmc_6b":
            hp.protvec_dim = 2560
    
    if args.cuda is not None:
        hp.cuda = args.cuda
    
    if args.dataset is not None:
        hp.set_dataset(args.dataset)
    else:
        if args.use_esmc is not None or args.esmc_model is not None:
            hp.set_dataset(hp.dataset)
    
    if args.running_set is not None:
        hp.running_set = args.running_set.replace('_', '-')
    if args.epochs is not None:
        hp.Epoch = args.epochs
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
    if args.num_experts is not None:
        hp.num_experts = args.num_experts
    if args.top_k is not None:
        hp.top_k = args.top_k
    if args.moe_noise_std is not None:
        hp.moe_noise_std = args.moe_noise_std
    if args.load_balance_weight is not None:
        hp.load_balance_weight = args.load_balance_weight

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    print(f"=" * 60)
    print(f"Training Fold {fold_i}/{hp.kfold-1}")
    print(f"Dataset: {hp.dataset}-{hp.running_set}") 
    print(f"ESM Model: {'ESM-C-' + hp.esmc_model if hp.use_esmc else 'ESM2'} (dim={hp.protvec_dim})")
    print(f"MoE: num_experts={hp.num_experts}, top_k={hp.top_k}, noise={hp.moe_noise_std}, lb_weight={hp.load_balance_weight}")
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={hp.cuda})")
    print(f"Pretrain-{hp.mol2vec_dir}")
    print(f"Pretrain-{hp.protvec_dir}")
    print(f"=" * 60)
    
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_config = {
            'dataset': hp.dataset,
            'running_set': hp.running_set,
            'fold': fold_i,
            'epochs': hp.Epoch,
            'batch_size': hp.Batch_size,
            'learning_rate': hp.Learning_rate,
            'max_patience': hp.max_patience,
            'cuda_device': hp.cuda,
            'use_esmc': hp.use_esmc,
            'esmc_model': hp.esmc_model if hp.use_esmc else None,
            'protvec_dim': hp.protvec_dim,
        }
        
        esm_name = f"esmc-{hp.esmc_model}" if hp.use_esmc else "esm2"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{hp.dataset}-{hp.running_set}-{esm_name}-fold{fold_i}",
            config=wandb_config,
            tags=[hp.dataset, hp.running_set, esm_name, f'fold{fold_i}'],
            reinit=True
        )
        print(f"Weights & Biases initialized: {args.wandb_project}")
    else:
        print("Weights & Biases logging disabled")
    
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    if fold_i < 0 or fold_i >= hp.kfold:
        raise ValueError(f"Fold index must be between 0 and {hp.kfold-1}, got {fold_i}")
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    train_dir = os.path.join(dataset_root, f'fold_{fold_i}_train.csv')
    valid_dir = os.path.join(dataset_root, f'fold_{fold_i}_valid.csv')
    test_dir = os.path.join(dataset_root, f'fold_{fold_i}_test.csv')
    
    print(f"Loading fold {fold_i} data...")
    print(f"  Train: {train_dir}")
    print(f"  Valid: {valid_dir}")
    print(f"  Test:  {test_dir}")
    
    train_set = CustomDataSet(pd.read_csv(train_dir, sep=','), hp)
    valid_set = CustomDataSet(pd.read_csv(valid_dir, sep=','), hp)
    test_set = CustomDataSet(pd.read_csv(test_dir, sep=','), hp)
    train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    print(f"Dataset loaded: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test samples")

    model = nn.DataParallel(CAMDTA(hp, device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    criterion = F.mse_loss

    train_log = []     
    best_valid_mse = float('inf')
    patience = 0    
    
    timestamp = hp.current_time
    model_fromTrain = f'./savemodel/{hp.dataset}-{hp.running_set}-fold{fold_i}-{timestamp}.pth'
    
    os.makedirs('./savemodel', exist_ok=True)
    
    print(f"Model will be saved to: {model_fromTrain}")
             
    for epoch in range(1, hp.Epoch + 1):    
        if hasattr(model.module, 'reset_usage_stats'):
            model.module.reset_usage_stats()
        
        model.train()
        pred = []
        label = []
        total_load_balance_loss = 0.0
        num_batches = 0
        for batch_data in train_dataset_load:
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data                    
            predictions, gate_info = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True)
            pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
            label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()            
            
            loss = criterion(predictions.squeeze(), affinity)
            
            load_balance_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])
            total_load_balance_loss += load_balance_loss.item()
            num_batches += 1
            
            total_loss = loss + hp.load_balance_weight * load_balance_loss
            
            total_loss.backward()                
            optimizer.step()
            optimizer.zero_grad()                                             
        pred = np.array(pred)
        label = np.array(label)
        mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(pred, label)
        train_log.append([mse_value, rmse_value, ci, r2, pearson_value, spearman_value])
        
        moe_stats = model.module.get_expert_usage_stats() if hasattr(model.module, 'get_expert_usage_stats') else None
        avg_load_balance_loss = total_load_balance_loss / num_batches if num_batches > 0 else 0
        
        print(f'Traing Log at fold-{fold_i} epoch-{epoch}: mse-{mse_value}, rmse-{rmse_value}, r2-{r2}')
        if moe_stats:
            print(f'  MoE Stats: usage_rate={moe_stats["expert_usage_rate"]}, entropy={moe_stats["usage_entropy"]:.4f}, dominant_expert={moe_stats["dominant_expert"]}, load_balance_loss={avg_load_balance_loss:.4f}')
        
        if hasattr(model.module, 'adaptive_update'):
            adjustments = model.module.adaptive_update(epoch, hp.Epoch, moe_stats)
            if 'load_balance_weight' in adjustments:
                hp.load_balance_weight = adjustments['load_balance_weight']
            print(f'  MoE Adaptive: noise={adjustments.get("noise_std", "N/A"):.4f}, lb_weight={adjustments.get("load_balance_weight", "N/A"):.4f}, lb_adj={adjustments.get("lb_adjustment", "N/A")}')
        
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/mse': mse_value,
                'train/rmse': rmse_value,
                'train/ci': ci,
                'train/r2': r2,
                'train/pearson': pearson_value,
                'train/spearman': spearman_value,
            }
            if moe_stats:
                log_dict['moe/load_balance_loss'] = avg_load_balance_loss
                log_dict['moe/usage_entropy'] = moe_stats['usage_entropy']
                log_dict['moe/usage_std'] = moe_stats['usage_std']
                log_dict['moe/dominant_expert'] = moe_stats['dominant_expert']
                for i, rate in enumerate(moe_stats['expert_usage_rate']):
                    log_dict[f'moe/expert_{i}_usage'] = rate
            if hasattr(model.module, 'adaptive_update'):
                log_dict['moe/adaptive_noise_std'] = adjustments.get('noise_std', 0)
                log_dict['moe/adaptive_lb_weight'] = adjustments.get('load_balance_weight', 0)
            wandb.log(log_dict)
        
        mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)   
        print(f'Valid at fold-{fold_i}: mse-{mse}')
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'valid/mse': mse,
                'valid/rmse': rmse,
                'valid/r2': r2,
                'valid/pearson': pearson,
                'valid/spearman': spearman,
            })
        
        if mse < best_valid_mse:
            patience = 0
            best_valid_mse = mse
            torch.save(model.state_dict(), model_fromTrain)
            print(f'Update best_mse, Valid at fold-{fold_i} epoch-{epoch}: mse-{mse}, rmse-{rmse}, ci-{ci}, r2-{r2}, pearson-{pearson}, spearman-{spearman}')
            
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'best_valid/mse': mse,
                    'best_valid/rmse': rmse,
                    'best_valid/r2': r2,
                    'best_valid/pearson': pearson,
                    'best_valid/spearman': spearman,
                })
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f'Traing stop at epoch-{epoch}, model save at-{model_fromTrain}')
                break   
             
    log_dir = f"./log/{timestamp}-{hp.dataset}-{hp.running_set}-fold{fold_i}.csv"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    with open(log_dir, "w+")as f:
        writer = csv.writer(f)
        writer.writerow(["mse", "rmse", "ci", "r2", 'pearson', 'spearman'])
        for r in train_log:
            writer.writerow(r)
    print(f'Save log over at {log_dir}')

    print(f"\n{'='*60}")
    print(f"Testing fold {fold_i} with best model...")
    print(f"{'='*60}")
    predModel = nn.DataParallel(CAMDTA(hp, device))
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel = predModel.to(device)    
    mse, rmse, ci, r2, pearson, spearman = test(predModel, test_dataset_load, is_valid=False)
    print(f'Test at fold-{fold_i}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n')
    
    if use_wandb:
        wandb.log({
            'test/mse': mse,
            'test/rmse': rmse,
            'test/ci': ci,
            'test/r2': r2,
            'test/pearson': pearson,
            'test/spearman': spearman,
        })
        wandb.summary['final_test_mse'] = mse
        wandb.summary['final_test_rmse'] = rmse
        wandb.summary['final_test_ci'] = ci
        wandb.summary['final_test_r2'] = r2
        wandb.summary['final_test_pearson'] = pearson
        wandb.summary['final_test_spearman'] = spearman
    
    fold_result_file = f'./log/Test-{hp.dataset}-{hp.running_set}-fold{fold_i}-{timestamp}.csv'
    fold_result = pd.DataFrame({
        'fold': [fold_i],
        'mse': [mse], 
        'rmse': [rmse], 
        'ci': [ci], 
        'r2': [r2], 
        'pearson': [pearson], 
        'spearman': [spearman]
    })
    fold_result.to_csv(fold_result_file, index=False)
    print(f"Fold {fold_i} results saved to: {fold_result_file}")
    print(f"{'='*60}")
    print(f"Training fold {fold_i} completed successfully!")
    print(f"{'='*60}")
    
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished")
