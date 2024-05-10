import os
import time
import warnings
from datetime import timedelta

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from collections import defaultdict
from tqdm import tqdm
from einops import rearrange
from IPython.display import clear_output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def beautiful_int(i):
    i = str(i)
    return ",".join(reversed([i[max(j, 0):j+3] for j in range(len(i) - 3, -3, -3)]))

def model_num_params(model, verbose_all=True, verbose_only_learnable=False):
    """
    Считаем общее число параметров в нашей модели
    """
    sum_params = 0
    sum_learnable_params = 0
    submodules = defaultdict(lambda : [0, 0])
    for name, param in model.named_parameters():
        num_params = np.prod(param.shape)
        if verbose_all or (verbose_only_learnable and param[1].requires_grad):
            print(
                colored(
                    '{: <62} ~  {: <9} params ~ grad: {}'.format(
                        name,
                        beautiful_int(num_params),
                        param.requires_grad,
                    ),
                    {True: "green", False: "red"}[param[1].requires_grad],
                )
            )
        sum_params += num_params
        sm = name.split(".")[0]
        submodules[sm][0] += num_params
        if param.requires_grad:
            sum_learnable_params += num_params
            submodules[sm][1] += num_params
    print(
        f'\nIn total:\n  - {beautiful_int(sum_params)} params\n  - {beautiful_int(sum_learnable_params)} learnable params'
    )
    
    for sm, v in submodules.items():
        print(
            f"\n . {sm}:\n .   - {beautiful_int(submodules[sm][0])} params\n .   - {beautiful_int(submodules[sm][1])} learnable params"
        )
    return sum_params, sum_learnable_params

def create_model_and_optimizer(model_class, model_params, lr=1e-5, beta1=0.9, beta2=0.999, device=get_device()):
    model = model_class(**model_params)
    model = model.to(device)
    
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
    
    optimizer = torch.optim.Adam(params, lr, [beta1, beta2])
    return model, optimizer


def train(model, optimizer, loader, criterion, device=get_device()):
    """
    Train iteration
    """
    model.train()
    losses_tr = []
    for seq_batch in tqdm(loader):
        seq_batch = seq_batch.to(device)
        optimizer.zero_grad()
        
        logits = model(seq_batch[:, 0:-1, ...])
        target = seq_batch[:, 1:None, 0]


        loss = criterion(rearrange(logits, "bs seq vocab -> (bs seq) vocab"), 
                         rearrange(target, "bs seq -> (bs seq)"))

        
        loss.backward()
        optimizer.step()
        losses_tr.append(loss.item())
    
    return model, optimizer, np.mean(losses_tr)


def val(model, loader, criterion, metric_names=None, device=get_device()):
    """
    Val iteration
    """
    model.eval()
    losses_val = []
    if metric_names is not None:
        metrics = defaultdict(list)
    with torch.no_grad():
        for seq_batch in tqdm(loader):
            seq_batch = seq_batch.to(device)
            
            logits = model(seq_batch[:, 0:-1, ...])
            target = seq_batch[:, 1:None, 0]  # take 1st feature in feature_list as target (mcc)
            
            pred = rearrange(logits, "bs seq vocab -> (bs seq) vocab")
            target = rearrange(target, "bs seq -> (bs seq)")


            loss = criterion(pred, target)
            losses_val.append(loss.item())
            
            # Можете добавить сюда любые метрики, которые хочется 
            if metric_names is not None:
                if 'accuracy' in metric_names:
                    preds = torch.argsort(pred, dim=-1, descending=True)
                    for k in metric_names["accuracy"]["top"]:
                        metrics[f'accuracy ~ top#{k}'].append(
                            np.mean([target[i] in preds[i, :k] for i in range(target.shape[0])])
                        )

        if metric_names is not None:
            for name in metrics:
                metrics[name] = np.mean(metrics[name])
    
    return np.mean(losses_val), metrics if metric_names else None

def learning_loop(
    model,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    scheduler=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    draw_every=1,
    separate_show=False,
    model_name=None,
    chkp_folder="./chkps",
    metric_names=None,
    device=get_device(),
):
    """
    Epoch training and validation
    """

    # Выбираем куда будем сохранять модель
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f'model#{num_starts}'
    else:
        if "#" not in model_name:
            model_name += "#0"
    changed = False
    while os.path.exists(os.path.join(chkp_folder, model_name + '.pt')):
        model_name, ind = model_name.split("#")
        model_name += f"#{int(ind) + 1}"
        changed = True
    if changed:
        warnings.warn(f"Selected model_name was used already! To avoid possible overwrite - model_name changed to {model_name}")
        
    # Инициализируем переменные
    losses = {'train': [], 'val': []}
    lrs = []
    best_val_loss = np.Inf
    if metric_names is not None:
        metrics = defaultdict(list)
    start_time = time.monotonic()

    # Цикл обучения
    for epoch in range(1, epochs+1):
        print(f'#{epoch}/{epochs}:')

        lrs.append(get_lr(optimizer))
        
        model, optimizer, loss = train(model, optimizer, train_loader, criterion, device=device)
        losses['train'].append(loss)

        # Каждые val_every эпох проводим валидацию
        if not (epoch % val_every):
            loss, metrics_ = val(model, val_loader, criterion, metric_names=metric_names, device=device)
            losses['val'].append(loss)
            if metrics_ is not None:
                for name, value in metrics_.items():
                    metrics[name].append(value)
            
            # Сохраняем лучшую по валидации модель
            if loss < best_val_loss:
                if not os.path.exists(chkp_folder):
                    os.makedirs(chkp_folder)
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'losses': losses,
                    },
                    os.path.join(chkp_folder, model_name + '.pt'),
                )
                best_val_loss = loss
            
            # Шаг шедулера
            if scheduler:
                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)

        # Каждые draw_every эпох рисуем графики
        if not (epoch % draw_every):
            clear_output(True)
            ww = 3 if separate_show else 2
            ww_metrics = 0
            if metric_names is not None:
                plot_ids_ = [
                    [key, metric_meta.get("plot_id", 1), metric_meta]
                    for key, metric_meta
                    in metric_names.items()
                ]
                ww_metrics = len(set(el[1] for el in plot_ids_))
                assert all(isinstance(el[1], int) for el in plot_ids_)
                assert all(el[1] <= ww_metrics for el in plot_ids_)
                assert all(el[1] >= 1 for el in plot_ids_)
                
                plot_ids = defaultdict(list)
                for el in plot_ids_:
                    plot_ids[el[1]].append((el[0], el[2]))
                
            fig, ax = plt.subplots(1, ww + ww_metrics, figsize=(30, 10))
            fig.suptitle(f'#{epoch}/{epochs} ~ {timedelta(seconds=time.monotonic() - start_time)}')

            plt.subplot(1, ww + ww_metrics, 1)
            plt.plot(losses['train'], 'r.-', label='train')
            if separate_show:
                plt.title('loss on train')
                plt.legend()
            plt.grid()

            if separate_show:
                plt.subplot(1, ww + ww_metrics, 2)
                plt.title('loss on validation')
                plt.grid()
            else:
                plt.title('losses')
            plt.plot(losses['val'], 'g.-', label='val')
            plt.legend()
            
            plt.subplot(1, ww + ww_metrics, ww)
            plt.title('learning rate')
            plt.plot(lrs, 'g.-', label='lr')
            plt.yscale("log")
            plt.legend()
            plt.grid()
            
            if metric_names is not None:
                for plot_id, keys_meta in plot_ids.items():
                    aggregated_meta = {}
                    plt.subplot(1, ww + ww_metrics, ww + plot_id)
                    if len(keys_meta) > 1:
                        plt.title(f'additional metrics #{plot_id}')
                    elif len(keys_meta) == 1:
                        plt.title(keys_meta[0][0])
                    for key_meta in keys_meta:
                        key, meta = key_meta
                        for meta_key in ["yscale"]:
                            if meta_key in meta:
                                assert meta_key not in aggregated_meta, f"Bad meta data '{meta_key}' doubled inside one plot_id ({plot_id})"
                                aggregated_meta[meta_key] = meta[meta_key]
                        for name in metrics:
                            if key in name:
                                plt.plot(metrics[name], '.-', label=f"{name} (last: {metrics[name][-1]: 0.2f})")
                    plt.yscale(aggregated_meta.get("yscale", "linear"))
                    plt.legend()
                    plt.grid()
            plt.show()
        
        # early_stopping - останавливаем обучение, если LR упал ниже min_lr
        if min_lr and get_lr(optimizer) <= min_lr:
            print(f'Learning process ended with early stop after epoch {epoch}')
            break
    
    return model, optimizer, losses, metrics if metric_names is not None else None


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Косинусный шедулер с warmup
    """
    def __init__(self, optimizer, warmup, max_epoch, min_lr=1e-9):
        self.warmup = warmup
        self.max_num_iters = max_epoch
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.min_lr]
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def train_pipeline(model_class, model_name, dataloader_train, dataloader_val, feature2vocab_size, hidden_dim=128, epochs=10):
    """
    Общая функция пайплайна тренировки модели (для простоты вызова в ноутбуке)
    """
    device = get_device()

    # Убедитесь что всё сработало и создалось нормально и без ошибок
    model, optimizer = create_model_and_optimizer(
        model_class=model_class, # BertSequenceModel
        model_params=dict(hidden_dim=hidden_dim, feature2vocab_dict=feature2vocab_size),
        lr=3e-4,
    )
    sum_params, sum_learnable_params = model_num_params(model, verbose_all=False)

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup=epochs // 10,
        max_epoch=epochs,
        min_lr=1e-7,
    )

    criterion = nn.CrossEntropyLoss()

    model, optimizer, losses, metrics = learning_loop(
        model = model,
        optimizer = optimizer,
        train_loader = dataloader_train,
        val_loader = dataloader_val,
        criterion = criterion,
        scheduler = scheduler,
        epochs = epochs,
        min_lr = 1e-8,
        val_every = 1,
        draw_every = 1,
        separate_show = False,
        metric_names = {
            "accuracy": {"top": [1, 5], "plot_id": 1},
        },
        chkp_folder = "./chkp",
        model_name = model_name,
        device = device
    )
    return metrics
