import torch, math, argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef
from torch.nn import BCELoss
from torch_geometric.data import DataLoader
from Dataset import MolNet
from Dataset_pre import MolNet_pre
from model import GFE
from utils import get_logger, metrics_c, metrics_r, set_seed, load_data, LogCoshLoss
from rdkit.Chem.SaltRemover import SaltRemover
import hyperopt
from hyperopt import fmin, hp, Trials
from hyperopt.early_stop import no_progress_loss
import warnings
import numpy as np
import csv


warnings.filterwarnings("ignore")
remover = SaltRemover()
bad = ['He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']
_use_shared_memory = True
torch.backends.cudnn.benchmark = True


def training(model, train_loader, optimizer, loss_f, metric, task, device, mean, stds):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([]); tars = torch.Tensor([])
    model.train()
    if task == 'clas':
        # loss_f = nn.BCEWithLogitsLoss().to(device)
        for data in train_loader:

            if data.y.size()[0] > 1:
                y = data.y.to(device)
                logits = model(data)

                loss = loss_f(logits.squeeze(), y.squeeze())
                loss_record += float(loss.item())
                record_count += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
                optimizer.step()

                pred = logits.detach().cpu()
                preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
        clas = preds > 0.5
        acc, f1, pre, rec, auc, mcc = metric(clas.squeeze().numpy(), preds.squeeze().numpy(), tars.squeeze().numpy())
    else:
        for data in train_loader:
            if data.y.size()[0] > 1:
                y = data.y.to(device)
                
                y_ = (y - mean) / (stds+1e-5)
                logits = model(data)
                
                loss = loss_f(logits.squeeze(), y_.squeeze())
                loss_record += float(loss.item())
                record_count += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
                optimizer.step()

                pred = logits.detach().cpu()
                pred = pred*stds+mean
                preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
        acc, f1, pre, rec, auc = metric(preds.squeeze().numpy(), tars.squeeze().numpy())

    epoch_loss = loss_record / record_count
    return epoch_loss, acc, f1, pre, rec, auc, mcc


def testing(model, test_loader, loss_f, metric, task, device, mean, stds, resu):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([]);
    tars = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        if task == 'clas':
            for data in test_loader:
                # loss_f = nn.BCEWithLogitsLoss().to(device)
                if data.y.size()[0] > 1:
                    y = data.y.to(device)
                    logits = model(data)

                    loss = loss_f(logits.squeeze(), y.squeeze())
                    loss_record += float(loss.item())
                    record_count += 1

                    pred = logits.detach().cpu()
                    preds = torch.cat([preds, pred], 0);
                    tars = torch.cat([tars, y.cpu()], 0)
            clas = preds > 0.5
            acc, f1, pre, rec, auc, mcc= metric(clas.squeeze().numpy(), preds.squeeze().numpy(), tars.squeeze().numpy())
        else:
            for data in test_loader:
                if data.y.size()[0] > 1:
                    y = data.y.to(device)

                    y_ = (y - mean) / (stds + 1e-5)
                    logits = model(data)

                    loss = loss_f(logits.squeeze(), y_.squeeze())
                    loss_record += float(loss.item())
                    record_count += 1

                    pred = logits.detach().cpu()
                    pred = pred * stds + mean
                    preds = torch.cat([preds, pred], 0);
                    tars = torch.cat([tars, y.cpu()], 0)
            acc, f1, pre, rec, auc = metric(preds.squeeze().numpy(), tars.squeeze().numpy())

    epoch_loss = loss_record / record_count
    if resu:
        return epoch_loss, acc, f1, pre, rec, auc, preds, tars, mcc
    else:
        return epoch_loss, acc, f1, pre, rec, auc, mcc


def predicting(model, preloader, task, device, mean, stds):
    preds = torch.Tensor([])
    model.eval()
    results = []
    smis = []
    with torch.no_grad():
        if task == 'clas':
            for data in preloader:
                batch_smis = data.smi
                logits = model(data)

                pred = logits.detach().cpu()
                preds = torch.cat([preds, pred], 0)
                # 对当前batch的预测结果进行处理
                batch_clas = pred > 0.5
                batch_clas = batch_clas.squeeze().int().tolist()
                # 确保batch_clas是列表的列表
                if not isinstance(batch_clas, list):
                    batch_clas = [[batch_clas]]  # 转换为二维列表
                elif isinstance(batch_clas[0], list) and len(batch_clas[0]) == 1:
                    # 如果是[[0], [1], ...]格式，展平为[0, 1, ...]
                    batch_clas = [p[0] for p in batch_clas]
                # 更新smis和results
                smis.extend(batch_smis)
                results.extend(zip(batch_smis, batch_clas))
        else:
            for data in preloader:
                if data.y.size()[0] > 1:
                    batch_smis = data.smi

                    logits = model(data)

                    pred = logits.detach().cpu()
                    pred = pred * stds + mean
                    preds = torch.cat([preds, pred], 0);
                    batch_reg = batch_reg.squeeze().int().tolist()
                    # 确保batch_reg是一维列表
                    if not isinstance(batch_reg, list):
                        batch_reg = [batch_reg]  # 单样本转为列表
                    elif isinstance(batch_reg[0], list) and len(batch_reg[0]) == 1:
                        batch_reg = [p[0] for p in batch_reg]
                    smis.extend(batch_smis)
                    results.extend(zip(batch_smis, batch_reg))

        # 写入CSV文件
    with open('predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['SMILES', 'Prediction'])  # 固定两列表头

        # 写入数据，确保每个SMILES对应一个预测值
        for smiles, prediction in results:
            writer.writerow([smiles, prediction])

            # acc, f1, pre, rec, auc = metric(clas.squeeze().numpy(), preds.squeeze().numpy(), tars.squeeze().numpy())
    #     else:
    #         for data in test_loader:
    #             if data.y.size()[0] > 1:
    #                 y = data.y.to(device)
    #
    #                 y_ = (y - mean) / (stds+1e-5)
    #                 logits = model(data)
    #
    #                 loss = loss_f(logits.squeeze(), y_.squeeze())
    #                 loss_record += float(loss.item())
    #                 record_count += 1
    #
    #                 pred = logits.detach().cpu()
    #                 pred = pred*stds+mean
    #                 preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
    #         acc, f1, pre, rec, auc = metric(preds.squeeze().numpy(), tars.squeeze().numpy())
    #
    # epoch_loss = loss_record / record_count
    # if resu:
    #     return epoch_loss, acc, f1, pre, rec, auc, preds, tars
    # else:
    #     return epoch_loss, acc, f1, pre, rec, auc





def predict(tasks, task, dataset, device, seed, batch_size, logger, attn_head, output_dim, attn_layers, dropout,pretrain, mean, stds, D):
    logger.info('Dataset: {}  task: {}  testing:'.format(dataset, task))
    d_k = round(output_dim / attn_head)
    if seed is not None:
        set_seed(seed)
    model = GFE(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
    state_dict = torch.load(pretrain)
    model.load_state_dict(state_dict)

    data = MolNet_pre(root='./dataset', dataset=dataset)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
    if task == 'clas':
        predicting(model, loader, task, device, mean, stds)
        logger.info('Predictions saved to predictions.csv')
    else:
        loss_f = LogCoshLoss().to(device)
        metric = metrics_r(mean_absolute_error, mean_squared_error, r2_score)
        loss, mae, rmse, r2, _, _, preds, tars = predicting(model, loader, loss_f, metric, task, device, mean, stds,
                                                             True)
        logger.info(
                'Dataset: {}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {:.4f} test_r2: {:.4f}'.format(dataset,
                                                                                                             loss, mae,
                                                                                                             rmse, r2))
        results = {
                'test_loss': loss,
                'test_mae': mae,
                'test_rmse': rmse,
                'test_r2': r2,
                'prediction': preds,
                'target': tars
        }
        np.save('log/Result' + moldata + '_test.npy', results, allow_pickle=True)


def main(tasks, task, dataset, device, train_epoch, seed,
         fold, batch_size, rate, scaffold, modelpath, logger, lr, attn_head,
         output_dim, attn_layers, dropout, mean, stds, D, met, savem):
    logger.info('Dataset: {}  task: {}  train_epoch: {}'.format(dataset, task, train_epoch))
    d_k, seed_ = round(output_dim/attn_head), seed

    fold_result = [[], []]
    if task == 'clas':
        loss_f = BCELoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef)

        for fol in range(1, fold+1):
            best_val_acc, best_test_acc = 0., 0.  # 更改变量名
            if seed is not None:
                seed_ = seed + fol-1
                set_seed(seed_)
            model = GFE(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            train_loader, valid_loader, test_loader = load_data(dataset, batch_size, rate[0], rate[1], 0, task, scaffold, seed_)
            logger.info('Dataset: {}  Fold: {:<4d}'.format(dataset, fol))

            for i in range(1, train_epoch + 1):
                train_loss, train_acc, train_f1, train_pre, train_rec, train_auc, train_mcc = training(model,
                                                                                                       train_loader,
                                                                                                       optimizer,
                                                                                                       loss_f, metric,
                                                                                                       task, device,
                                                                                                       mean, stds)
                valid_loss, valid_acc, valid_f1, valid_pre, valid_rec, valid_auc, valid_mcc = testing(model,
                                                                                                      valid_loader,
                                                                                                      loss_f, metric,
                                                                                                      task, device,
                                                                                                      mean, stds, False)

                logger.info(
                    'Dataset: {}  Epoch: {:<3d}  train_loss: {:.4f}  train_acc: {:.4f}  train_f1: {:.4f}  train_auc: {:.4f}  train_pre: {:.4f}  train_rec: {:.4f}  train_mcc: {:.4f}'.format(
                        dataset, i, train_loss, train_acc, train_f1, train_auc, train_pre, train_rec, train_mcc))
                logger.info(
                    'Dataset: {}  Epoch: {:<3d}  valid_loss: {:.4f}  valid_acc: {:.4f}  valid_f1: {:.4f}  valid_auc: {:.4f}  valid_pre: {:.4f}  valid_rec: {:.4f}  valid_mcc: {:.4f}'.format(
                        dataset, i, valid_loss, valid_acc, valid_f1, valid_auc, valid_pre, valid_rec, valid_mcc))

                # 修改条件判断为valid_acc
                if valid_acc > best_val_acc:
                    best_val_acc = valid_acc
                    if savem:
                        # 文件名使用valid_acc
                        model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_acc, 4))
                        torch.save(model.state_dict(), model_save_path)
                    test_loss, test_acc, test_f1, test_pre, test_rec, test_auc, test_mcc = testing(model, test_loader,
                                                                                                   loss_f, metric, task,
                                                                                                   device, mean, stds,
                                                                                                   False)
                    logger.info(
                        'Dataset: {}  Epoch: {:<3d}  test__loss: {:.4f}  test__acc: {:.4f}  test__f1: {:.4f}  test__auc: {:.4f}  test__pre: {:.4f}  test__rec: {:.4f}  test__mcc: {:.4f}'.format(
                            dataset, i, test_loss, test_acc, test_f1, test_auc, test_pre, test_rec, test_mcc))
                    best_test_auc = test_auc  # 记录测试acc

            fold_result[0].append(best_val_acc)  # 保存验证acc
            fold_result[1].append(best_test_acc)  # 保存测试acc
            logger.info('Dataset: {} Fold: {} best_val_acc: {:.4f}  best_test_acc: {:.4f}'.format(dataset, fol, best_val_acc, best_test_acc))
        logger.info('Dataset: {} Fold result: {}'.format(dataset, fold_result))
        return fold_result
    else:
        loss_f = LogCoshLoss().to(device)
        metric = metrics_r(mean_absolute_error, mean_squared_error, r2_score)

        for fol in range(1, fold + 1):
            best_val_rmse, best_test_rmse = 9999., 9999.
            if seed is not None:
                seed_ = seed + fol - 1
                set_seed(seed_)
            model = GFE(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

            train_loader, valid_loader, test_loader = load_data(dataset, batch_size, rate[0], rate[1], 0, task,
                                                                scaffold, seed_)
            logger.info('Dataset: {}  Fold: {:<4d}'.format(moldata, fol))

            for i in range(1, train_epoch + 1):
                train_loss, train_mae, train_rmse, train_r2, _, _ = training(model, train_loader, optimizer, loss_f,
                                                                             metric, task, device, mean, stds)
                valid_loss, valid_mae, valid_rmse, valid_r2, _, _ = testing(model, valid_loader, loss_f, metric, task,
                                                                            device, mean, stds, False)
                logger.info('Dataset: {}  Epoch: {:<3d}  train_loss: {:.4f}  train_mae: {:.4f}  train_rmse: {:.4f} '
                            'train_r2: {:.4f}'.format(dataset, i, train_loss, train_mae, train_rmse, train_r2))
                logger.info('Dataset: {}  Epoch: {:<3d}  valid_loss: {:.4f}  valid_mae: {:.4f}  valid_rmse: {:.4f} '
                            'valid_r2: {:.4f}'.format(dataset, i, valid_loss, valid_mae, valid_rmse, valid_r2))

                if met == 'rmse':
                    if valid_rmse < best_val_rmse:
                        best_val_rmse = valid_rmse
                        if savem:
                            model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_rmse, 4))
                            torch.save(model.state_dict(), model_save_path)
                        test_loss, test_mae, test_rmse, test_r2, _, _ = testing(model, test_loader, loss_f, metric,
                                                                                task, device, mean, stds, False)
                        logger.info('Dataset: {}  Epoch: {:<3d}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {'
                                    ':.4f} test_r2: {:.4f}'.format(dataset, i, test_loss, test_mae, test_rmse, test_r2))
                        best_test_rmse = test_rmse

                elif met == 'mae':
                    if valid_mae < best_val_rmse:
                        best_val_rmse = valid_mae
                        if savem:
                            model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_rmse, 4))
                            torch.save(model.state_dict(), model_save_path)
                        test_loss, test_mae, test_rmse, test_r2, _, _ = testing(model, test_loader, loss_f, metric,
                                                                                task, device, mean, stds, False)
                        logger.info('Dataset: {}  Epoch: {:<3d}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {'
                                    ':.4f} test_r2: {:.4f}'.format(dataset, i, test_loss, test_mae, test_rmse, test_r2))
                        best_test_rmse = test_mae
                else:
                    raise ValueError('regression metric must be rmse or mae')
            fold_result[0].append(best_val_rmse)
            fold_result[1].append(best_test_rmse)
            logger.info('Dataset: {} Fold: {} best_val_{}: {:.4f}  best_test_{}: {:.4f}'.format(dataset, fol, met,
                                                                                                best_val_rmse, met,
                                                                                                best_test_rmse))
        logger.info('Dataset: {} Fold result: {}'.format(dataset, fold_result))
        return fold_result


def test(tasks, task, dataset, device, seed, batch_size, logger,
         attn_head, output_dim, attn_layers, dropout, pretrain, mean, stds, D):
    logger.info('Dataset: {}  task: {}  testing:'.format(dataset, task))
    d_k = round(output_dim/attn_head)
    if seed is not None:
        set_seed(seed)
    model = GFE(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
    state_dict = torch.load(pretrain)
    model.load_state_dict(state_dict)

    data = MolNet(root='./dataset', dataset=dataset)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
    if task == 'clas':
        loss_f = BCELoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef)
        loss, acc, f1, pre, rec, auc, preds, tars, mcc = testing(model, loader, loss_f, metric, task, device, mean, stds, True)
        logger.info('Dataset: {}  test_loss: {:.4f}  test_acc: {:.4f}  test_f1: {:.4f}  test_auc: {:.4f}  test_pre: {:.4f}  test_rec: {:.4f}  test_mcc: {:.4f}'.format(dataset, loss, acc, f1, auc, pre, rec, mcc))
        results = {
            'test_loss': loss,
            'test_acc': acc,
            'test_f1': f1,
            'test_pre': pre,
            'test_rec': rec,
            'test_auc': auc,
            'test_mcc': mcc,
            'prediction': preds,
            'target': tars
        }
        np.save('log/Result'+moldata+'_test.npy', results, allow_pickle=True)
    else:
        loss_f = LogCoshLoss().to(device)
        metric = metrics_r(mean_absolute_error, mean_squared_error, r2_score)
        loss, mae, rmse, r2, _, _, preds, tars= testing(model, loader, loss_f, metric, task, device, mean, stds, True)
        logger.info('Dataset: {}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {:.4f} test_r2: {:.4f}'.format(dataset, loss, mae, rmse, r2))
        results = {
            'test_loss': loss,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2,
            'prediction': preds,
            'target': tars
        }
        np.save('log/Result'+moldata+'_test.npy', results, allow_pickle=True)


def psearch(params):
    logger.info('Optimizing Hyperparameters')
    fold_result = main(params['tasks'],params['task'],params['moldata'],params['device'],params['train_epoch'],params['seed'],params['fold'],params['batch_size'],params['rate'],params['scaffold'],params['modelpath'],params['logger'],params['lr'],params['attn_head'],params['output_dim'],params['attn_layers'],params['dropout'],params['mean'], params['std'], params['D'], params['metric'], False)
    if task == 'reg':
        valid_res = np.mean(fold_result[1])
    else:
        valid_res = -np.mean(fold_result[1])
    return valid_res


if __name__ == '__main__':
    # mode = 'train'
    mode = 'test'
    # mode = 'search'
    # mode = 'pre'
    moldata = 'ACE'
    task = 'clas'
    device = 'cuda:0'
    batch_size = 32
    train_epoch = 80
    valrate = 0.063
    testrate = 0.062
    fold = 3
    scaffold = False
    D = 16
    attn_head = 8
    attn_layers = 1
    dropout = 0.05
    lr = 0.0001
    output_dim = 256
    seed = 1
    pretrain = './log/checkpoint/ACE2origin/ACEclas_13_0.7823.pkl'
    metric = 'rmse'

    device = torch.device(device)
    rate = [valrate, testrate]

    device = torch.device(device)

    mean, std = 0, 1
    numtasks = 1
    logf = 'log/{}_{}_{}.log'.format(moldata, task, mode)
    modelpath = 'log/checkpoint/ACE2origin/'
    logger = get_logger(logf)
    
    moldata += task
    if mode == 'search':
        trials = Trials()
        if moldata in ['tox21', 'lipo', 'qm7']:
            batch_size = 256
        else:
            batch_size = 32
        if moldata == 'tox21':
            lrs = [1e-2, 5e-3, 1e-3]
        else:
            lrs = [1e-3, 5e-4, 1e-4]

        parm_space = {      # search space of param
            'tasks': numtasks,
            'task': task,
            'moldata': moldata,
            'mean': mean,
            'std': std,
            'device': device,
            'modelpath': modelpath,
            'logger': logger,
            'seed': seed,
            'fold': fold,
            'metric': metric,
            'rate': rate,
            'scaffold': scaffold,
            'train_epoch': train_epoch,
            'attn_head': hp.choice('attn_head', [4, 6, 8, 10]),
            'output_dim': hp.choice('output_dim', [128, 256]),
            'attn_layers': hp.choice('attn_layers', [1, 2, 3, 4]),
            'dropout': hp.choice('dropout', [0.05, 0.1]),
            'lr': hp.choice('lr', lrs),
            'D': hp.choice('D', [2, 4, 6, 8, 12, 16]),
            'batch_size': batch_size
            }
        param_mappings = {
            'attn_head': [4, 6, 8, 10],
            'output_dim': [128, 256],
            'attn_layers': [1, 2, 3, 4],
            'dropout': [0.05, 0.1],
            'lr': lrs,  # Placeholder for lr values. Please replace with actual values before running.
            'D': [2, 4, 6, 8, 12, 16]
            }
        best = fmin(fn=psearch, space=parm_space, algo=hyperopt.tpe.suggest, max_evals=100, trials=trials, early_stop_fn=no_progress_loss(50))
        best_values = {k: param_mappings[k][v] if k in param_mappings else v for k, v in best.items()}
        logger.info('Dataset {} Best Params: {}'.format(moldata, best_values))
        ys = [t['result']['loss'] for t in trials.trials]
        logger.info('Dataset {} Hyperopt Results: {}'.format(moldata, ys))

    elif mode == 'train':
        logger.info('Training')
        fold_result = main(numtasks, task, moldata, device, train_epoch, seed, fold, batch_size, rate,scaffold, modelpath, logger, lr, attn_head, output_dim,
                           attn_layers, dropout, mean, std, D, metric, True)
        print(fold_result)

    elif mode == 'test':
        assert (pretrain is not None)
        fold_result = test(numtasks, task, moldata, device, seed, batch_size, logger, attn_head,
                           output_dim, attn_layers, dropout, pretrain, mean, std, D)
    else:
        predict(numtasks, task, moldata, device, seed, batch_size, logger, attn_head,
                           output_dim, attn_layers, dropout, pretrain, mean, std, D)

