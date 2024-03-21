import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

from torch.utils.data import DataLoader

import os
import copy
import argparse
from tqdm import tqdm

from models.dsmil import FCLayer, BClassifier, MILNet

class EarlyStopper:
    def __init__(self, patience = 5, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
    
    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_loop(model, train_loader, optimizer, criterion, feats_size): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    train_loss = 0.

    for batch_idx, (data_batch, label_batch, _) in enumerate(train_loader):
        num_examples = len(data_batch)
        for i in range(num_examples):
            data = data_batch[i]
            label = label_batch[i]
            optimizer.zero_grad()
            data = data.to(device)
            ins_prediction, bag_prediction, _, _ = model(data.view(-1, feats_size))
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_label = torch.tensor([[label]]).to(device)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.type(torch.float32))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.type(torch.float32))       
            loss = 0.5*bag_loss + 0.5*max_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    train_loss /= len(train_loader)
   
    return train_loss

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def val_loop(model, val_loader, criterion, feats_size, num_classes): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_loss = 0.

    prob = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data_batch, label_batch, _) in enumerate(val_loader):
            num_examples = len(data_batch)
            for i in range(num_examples):
                data = data_batch[i]
                label = label_batch[i]
                data = data.to(device, non_blocking = True) 
                ins_prediction, bag_prediction, _, _ = model(data.view(-1,feats_size))
                max_prediction, _ = torch.max(ins_prediction, 0)
                bag_label_ext = torch.tensor([[label]]).to(device)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label_ext.type(torch.float32))
                max_loss = criterion(max_prediction.view(1, -1), bag_label_ext.type(torch.float32))
                loss = 0.5*bag_loss + 0.5*max_loss
                val_loss += loss.item()
                labels.extend([bag_label_ext.cpu().detach().squeeze(0).numpy()])
                prob.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])

        labels = np.array(labels)    
        prob = np.array(prob)
        auc_value, _, thresholds_optimal = multi_label_roc(labels, prob, num_classes, pos_label=1)

        if num_classes == 1:
            class_prediction_bag = copy.deepcopy(prob)
            class_prediction_bag[prob>=thresholds_optimal[0]] = 1
            class_prediction_bag[prob<thresholds_optimal[0]] = 0
            prob = class_prediction_bag
            labels = np.squeeze(labels)
        else:  
            for i in range(num_classes):
                class_prediction_bag = copy.deepcopy(prob[:, i])
                class_prediction_bag[prob[:, i]>=thresholds_optimal[i]] = 1
                class_prediction_bag[prob[:, i]<thresholds_optimal[i]] = 0
                prob[:, i] = class_prediction_bag
        bag_score = 0
        for i in range(0, len(val_loader)):
            try:
                bag_score = np.array_equal(labels[i], prob[i]) + bag_score       
            except:
                continue
        avg_score = bag_score / len(val_loader)

    return val_loss / len(val_loader), avg_score, auc_value, thresholds_optimal

def fit(model, loader, optimizer, scheduler, criterion, args):
    best_score = 0.0
    best_auc = 0.0
    best_model = None
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    for epoch in tqdm(range(1, args.num_epochs+1)):
        train_loss_bag = train_loop(model, loader['train'], optimizer, criterion, args.feats_size) 
        val_loss_bag, avg_score, aucs, thresholds_optimal = val_loop(model, loader['val'], criterion, args.feats_size, args.num_classes)
        
        if args.num_classes == 2:
            print('\r Epoch [%d/%d]  train loss: %.4f, val loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                    (epoch, args.num_epochs, train_loss_bag, val_loss_bag, avg_score, aucs[0], aucs[1]))
            
        elif args.num_classes == 1:
            print('\r Epoch [%d/%d]  train loss: %.4f, val loss: %.4f, average score: %.4f, auc: %.4f' % 
                    (epoch, args.num_epochs, train_loss_bag, val_loss_bag, avg_score, aucs[0]))
        
        scheduler.step()
        # current_score = (sum(aucs) + avg_score)/2

        if early_stopper.early_stop(val_loss_bag):
            break

        avg_auc = sum(aucs)/len(aucs)

        if val_loss_bag > train_loss_bag and avg_auc > best_auc:
            best_auc = avg_auc
            best_epoch = epoch
            best_model = model.state_dict()
            if args.num_classes == 2:
                thresh_opt_0, thresh_opt_1 = thresholds_optimal[0], thresholds_optimal[1]
            else:
                thresh_opt_0 = thresholds_optimal[0]
    
    if args.num_classes == 2:
        return avg_auc, best_epoch, best_model, thresh_opt_0, thresh_opt_1
    else:
        return best_auc, best_epoch, best_model, thresh_opt_0

def fit_model(model, loader, optimizer, scheduler, criterion, feats_size=256, 
              num_epochs=50, num_classes=3, use_validation=True):
    best_score = 0.0
    best_auc = 0.0
    best_epoch = -1
    best_model = None
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    for epoch in tqdm(range(1, num_epochs+1)):
        train_loss_bag = train_loop(model, loader['train'], optimizer, criterion, feats_size) 
        if use_validation:
            val_loss_bag, avg_score, aucs, thresholds_optimal = val_loop(model, loader['val'], criterion, feats_size, num_classes)
        else:
            val_loss_bag, avg_score, aucs, thresholds_optimal = [-1, -1, [-1,-1], -1]
        if num_classes == 2:
            print('\r Epoch [%d/%d]  train loss: %.4f, val loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                    (epoch, num_epochs, train_loss_bag, val_loss_bag, avg_score, aucs[0], aucs[1]))
            
        elif num_classes == 1:
            print('\r Epoch [%d/%d]  train loss: %.4f, val loss: %.4f, average score: %.4f, auc: %.4f' % 
                    (epoch, num_epochs, train_loss_bag, val_loss_bag, avg_score, aucs[0]))
        
        scheduler.step()
        # current_score = (sum(aucs) + avg_score)/2

        if use_validation and early_stopper.early_stop(val_loss_bag):
            break

        avg_auc = sum(aucs)/len(aucs)

        if use_validation and (val_loss_bag > train_loss_bag and avg_auc > best_auc):
            best_auc = avg_auc
            best_epoch = epoch
            best_model = model.state_dict()
            if num_classes == 2:
                thresh_opt_0, thresh_opt_1 = thresholds_optimal[0], thresholds_optimal[1]
            else:
                thresh_opt_0 = thresholds_optimal[0]
        else:
            thresh_opt_0, thresh_opt_1 = [-1, -1]
            best_model = model.state_dict()
    
    if num_classes == 2:
        return avg_auc, best_epoch, best_model, thresh_opt_0, thresh_opt_1
    else:
        return best_auc, best_epoch, best_model, thresh_opt_0
    
def train_dsmil_model(model, loader, num_runs=1, lr=5e-3, weight_decay=1e-4, num_epochs=50, num_classes = 3, 
                      feats_size=256,
                      folder_to_save=None, base_name = None, use_validation=True):
    aucs = []
    if folder_to_save is None:
        folder_to_save = os.getcwd()
    for i in tqdm(range(int(num_runs)), total = int(num_runs)):

        seed = np.random.randint(10000)

        torch.manual_seed(seed)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.5, 0.9), weight_decay= weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.000005)

        print(f"Experiment {i+1}/{num_runs}")

        if base_name is None:
            model_path = os.path.join(folder_to_save,  "seed_" + str(seed) + ".pt")
        else:
            model_path = os.path.join(folder_to_save,  base_name + "_seed_" + str(seed) + ".pt")
            
        if num_classes == 2:
            auc, best_epoch, best_model, thresh_opt_0, thresh_opt_1 = fit_model(model, loader, optimizer, scheduler, criterion, feats_size=feats_size,
                                                                                num_epochs=num_epochs, num_classes=num_classes, use_validation=use_validation)
            print('Average AUC', auc, 'Best Epoch', best_epoch, 'Seed:', seed, 'Thresholds: ', thresh_opt_0, '/', thresh_opt_1)
            torch.save({'model_state_dict': best_model, 'seed': seed, 'epoch': best_epoch, 'luad_thresh' : thresh_opt_0, 'lusc_thresh' : thresh_opt_1}, model_path)
        else:
            auc, best_epoch, best_model, thresh_opt_0 = fit_model(model, loader, optimizer, scheduler, criterion, feats_size=feats_size,
                                                                                num_epochs=num_epochs, num_classes=num_classes, use_validation=use_validation)
            print('AUC', auc, 'Best Epoch', best_epoch, 'Seed:', seed, 'Threshold: ', thresh_opt_0)
            torch.save({'model_state_dict': best_model, 'seed': seed, 'epoch': best_epoch, 'thresh' : thresh_opt_0}, model_path)
