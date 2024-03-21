import torch 
import torch.nn as nn

import numpy as np
from scipy.special import softmax

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
from skimage import exposure, transform
from PIL import Image

from torch.utils.data import DataLoader

from models.dsmil import FCLayer, BClassifier, MILNet

import os, glob
import copy
import json
import argparse

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

@torch.no_grad()
def test_loop(model, test_loader, thresh_luad, thresh_lusc, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_labels = []
    test_predictions = []
    for batch_idx, (data, bag_label, _) in enumerate(test_loader):
        data = data.to(device, non_blocking = True)
        ins_prediction, bag_prediction, _, _ = model(data.view(-1, args.feats_size))
        max_prediction, _ = torch.max(ins_prediction, 0)  
        bag_label_ext = torch.tensor([bag_label]).to(device)    
        test_labels.extend([bag_label_ext.cpu().detach().squeeze(0).numpy()])
        test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value = roc_auc_score(test_labels, test_predictions, multi_class='ovr')   

    return auc_value
    
def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i_classifier = FCLayer(in_size=args.feats_size, out_size=args.num_classes).to(device)
    b_classifier = BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v = 0, nonlinear = 1).to(device)
    return MILNet(i_classifier, b_classifier).to(device)

def histrogram(aucs, num_trainings):
    mean_auc = np.mean(aucs)
    min_int = mean_auc - 1.96*np.std(aucs) / np.sqrt(num_trainings)
    max_int = mean_auc + 1.96*np.std(aucs) / np.sqrt(num_trainings)
    plt.figure()
    plt.title('Mean AUC: {:.4f}, 95% confidence interval: [{:.4f}, {:.4f}]. # of exps: {:.0f}'.format(mean_auc, min_int, max_int, len(aucs)))
    plt.xlabel('AUC')
    plt.ylabel('Number of experiments')
    plt.hist(aucs)
    plt.show()

def save_json(list, output_path):
    out = open(output_path, 'w')
    out.write(json.dumps(list, indent = 2))
    out.close()

def test_dsmil(args, test_loader):

    model = load_model(args)
        
    if args.make_histogram:

        ckpts = [ckpt for ckpt in os.listdir(args.model_path) if os.path.isfile(os.path.join(args.model_path, ckpt))]
        auc_test = []   
        for ckpt in ckpts:
            model_path = os.path.join(args.model_path, ckpt)
            model.load_state_dict(torch.load(model_path)['model_state_dict']) 
            
            if args.num_classes == 1:
                thresh = torch.load(model_path)['thresh']
                thresh_luad = thresh_lusc = thresh
            else:
                thresh_luad = torch.load(model_path)['luad_thresh']
                thresh_lusc = torch.load(model_path)['lusc_thresh']
            auc = test_loop(model = model, test_loader = test_loader, thresh_luad = thresh_luad, thresh_lusc = thresh_lusc, args = args)
            auc_test.append(auc)

        histrogram(auc_test, len(auc_test))
    
    else:

        if args.num_classes == 1:
            model_path = args.model_path
            model.load_state_dict(torch.load(model_path)['model_state_dict']) 
            thresh = torch.load(model_path)['thresh']
            thresh_luad = thresh_lusc = thresh
        else:
            model_path = os.path.join(args.model_path, ckpt)
            thresh_luad = torch.load(model_path)['luad_thresh']
            thresh_lusc = torch.load(model_path)['lusc_thresh']
        auc = test_loop(model = model, test_loader = test_loader, thresh_luad = thresh_luad, thresh_lusc = thresh_lusc, args = args)
        # print('AUC = ', auc)
        return auc
    

def test_dsmil_model(model, test_loader, feats_size=256, num_classes=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_labels = []
    test_predictions = []
    for batch_idx, (data_batch, bag_label_batch, _) in enumerate(test_loader):
        num_examples = len(data_batch)
        for i in range(num_examples):
            data = data_batch[i]
            bag_label = bag_label_batch[i]
            data = data.to(device, non_blocking = True)
            ins_prediction, bag_prediction, _, _ = model(data.view(-1, feats_size))
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_label_ext = torch.tensor([bag_label]).to(device)    
            test_labels.extend([bag_label_ext.detach().cpu().squeeze(0).numpy()])
            test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().detach().cpu().numpy()])
            
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value = roc_auc_score(test_labels, test_predictions, multi_class='ovr')   

    return auc_value, test_labels, test_predictions