import numpy as np
from sklearn.metrics import roc_auc_score, auc, accuracy_score, precision_recall_curve, f1_score, balanced_accuracy_score, \
    recall_score, precision_score, matthews_corrcoef

from sklearn.metrics import hamming_loss
import os
import pandas as pd

def instances_overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    计算样本层面的整体评价指标
    """
    y_pred_cls = np.zeros_like(y_pred, dtype=np.int32)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别
    n, m = y_true.shape
    # Hamming Loss
    HLoss = hamming_loss(y_true, y_pred_cls)
    # Accuracy
    ACC = 0
    for i in range(n):
        ACC += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum((y_pred_cls[i] == 1) | (y_true[i] == 1)))
    ACC /= n
    # Precision
    Precision = 0
    for i in range(n):
        if (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) == 0): continue
        Precision += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_pred_cls[i] == 1) )
    Precision /= n
    # Recall
    Recall = 0
    for i in range(n):
        Recall += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_true[i] == 1))
    Recall /= n

    # Absolute ture
    AT = 0
    for i in range(n):
        if(np.all(y_pred_cls[i] == y_true[i])):
            AT += 1
    AT /= n
    df = pd.DataFrame({'HLoss': [HLoss], 'Accuracy': [ACC], 'Precision': [Precision], 'Recall': [Recall], 'Absolute true': [AT]})
    # if show:
    #     print(df)
    # if save is not None:
    #     df.to_csv(save)

    return df

def label_overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    计算macro和micro指标
    """

    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred >= threshold] = 1    # 预测类别

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []
    res_bacc = []

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        tp = np.sum(np.logical_and(y_t == pos, y_c == pos))
        tn = np.sum(np.logical_and(y_t == neg, y_c == neg))
        fp = np.sum(np.logical_and(y_t == neg, y_c == pos))
        fn = np.sum(np.logical_and(y_t == pos, y_c == neg))

        TP += tp
        TN += tn
        FP += fp
        FN += fn

        F1 = f1_score(y_t, y_c)
        ACC = accuracy_score(y_t, y_c)
        AUC = roc_auc_score(y_t, y_p)
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        AUPR = auc(recall, precision)
        BACC = balanced_accuracy_score(y_t, y_c)
        MCC = matthews_corrcoef(y_t, y_c)
        Recall = recall_score(y_t, y_c)
        Precision = precision_score(y_t, y_c)

        res_mcc.append(round(MCC, 3))
        res_acc.append(round(ACC, 3))
        res_auc.append(round(AUC, 3))
        res_aupr.append(round(AUPR, 3))
        res_recall.append(round(Recall, 3))
        res_precision.append(round(Precision, 3))
        res_f1.append(round(F1, 3))
        res_bacc.append(round(BACC, 3))

    ACC_micro = (TP + TN) / (TP + TN + FP + FN)
    MCC_micro = (TP * TN - FP * FN) / (np.sqrt(TP + FN) * np.sqrt(TP + FP) * np.sqrt(TN + FP) * np.sqrt(TN + FN))
    Precision_micro = TP / (TP + FP)
    Recall_micro = TP / (TP + FN)
    BACC_micro = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    AUC_micro = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
    AUPR_micro = 0
    F1_micro = f1_score(y_true, y_pred_cls, average='micro')

    ACC_macro = np.mean(res_acc)
    MCC_macro = np.mean(res_mcc)
    Precision_macro = np.mean(res_precision)
    Recall_macro = np.mean(res_recall)
    BACC_macro = np.mean(res_bacc)
    AUC_macro = np.mean(res_auc)
    AUPR_macro = np.mean(res_aupr)
    F1_macro = np.mean(res_f1)


    df = pd.DataFrame({'ACC': [ACC_macro, ACC_micro], 'BACC': [BACC_macro, BACC_micro],
                       'AUC': [AUC_macro, AUC_micro], 'MCC': [MCC_macro, MCC_micro],
                       'AUPR': [AUPR_macro, AUPR_micro], 'F1': [F1_macro, F1_micro],
                       'Precision': [Precision_macro, Precision_micro], 'Recall': [Recall_macro, Recall_micro]},
                      index= ['macro','micro'])
    # if show:
    #     print(df)
    # if save is not None:
    #     df.to_csv(save)

    return df

def binary_metrics(y_pred: np.array, y_true: np.array, class_names, threshold=0.5, save = None, show = True):
    """
    计算每一类的准确率，精度, 召回率, MCC
    Args:
        y_pred: 预测得分, [n_samlpes, n_class]
        y_true: 真实类别, [n_samlpes, n_class]
    """
    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred >= threshold] = 1    # 预测类别


    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []
    res_bacc = []

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        tp = np.sum(np.logical_and(y_t == pos, y_c == pos))
        tn = np.sum(np.logical_and(y_t == neg, y_c == neg))
        fp = np.sum(np.logical_and(y_t == neg, y_c == pos))
        fn = np.sum(np.logical_and(y_t == pos, y_c == neg))

        F1 = f1_score(y_t, y_c)
        ACC = accuracy_score(y_t, y_c)
        AUC = roc_auc_score(y_t, y_p)
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        AUPR = auc(recall, precision)
        BACC = balanced_accuracy_score(y_t, y_c)
        MCC = matthews_corrcoef(y_t, y_c)
        Recall = recall_score(y_t, y_c)
        Precision = precision_score(y_t, y_c)

        # if tp * tn - fp * fn == 0:
        #     MCC = 0
        # else:
        #     MCC = (tp * tn - fp * fn) / (np.sqrt(tp + fn) * np.sqrt(tp + fp) * np.sqrt(tn + fp) * np.sqrt(tn + fn))
        #
        # if tp == 0:
        #     Recall = 0
        #     Precision = 0
        # else:
        #     Recall = tp / (tp + fn)
        #     Precision = tp / (tp + fp)

        res_mcc.append(round(MCC, 4))
        res_acc.append(round(ACC, 4))
        res_auc.append(round(AUC, 4))
        res_aupr.append(round(AUPR, 4))
        res_recall.append(round(Recall, 4))
        res_precision.append(round(Precision, 4))
        res_f1.append(round(F1, 4))
        res_bacc.append(round(BACC, 4))

    df = pd.DataFrame({'ACC': res_acc, 'BACC': res_bacc,'AUC': res_auc, 'MCC': res_mcc, 'AUPR': res_aupr, 'F1': res_f1,
                       'Precision': res_precision, 'Recall': res_recall}, index=class_names)
    # print(df)
    # df.to_csv('new_save_model/ada_fl.csv')
    # exit(0)
    # if show:
    #     print(df)
    # if save is not None:
    #     df.to_csv(save)

    return df


def overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    综合评价多标签分类任务
    """
    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别


    HLoss = hamming_loss(y_true, y_pred_cls)

    # Calculate metrics globally by counting the total true positives,false negatives and false positives.
    F1_micro = f1_score(y_true, y_pred_cls, average='micro')

    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    F1_macro = f1_score(y_true, y_pred_cls, average='macro')

    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
    # This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.

    F1_weighted = f1_score(y_true, y_pred_cls, average='weighted')

    df = pd.DataFrame({'HLoss': [HLoss], 'F1_micro': [F1_micro], 'F1_macro': [F1_macro], 'F1_weighted': [F1_weighted]})
    # if show:
    #     print(df)

    # if save is not None:
    #     df.to_csv(save)

    return df


def weighted_binary_metrics(y_pred: np.array, y_true: np.array, class_names, thresholds , save = None, show = True):
    """
    根据每个类训练集正样本占比，设置阈值
    """
    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    y_pred_cls = np.zeros_like(y_pred)
    for i, pt in enumerate(class_names):
        y_pred_cls[:, i][y_pred[:, i] >= thresholds[pt]] = 1    # 预测类别

    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        tp = np.sum(np.logical_and(y_t == pos, y_c == pos))
        tn = np.sum(np.logical_and(y_t == neg, y_c == neg))
        fp = np.sum(np.logical_and(y_t == neg, y_c == pos))
        fn = np.sum(np.logical_and(y_t == pos, y_c == neg))

        F1 = f1_score(y_t, y_c)
        ACC = accuracy_score(y_t, y_c)
        AUC = roc_auc_score(y_t, y_p)
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        AUPR = auc(recall, precision)

        if tp * tn - fp * fn == 0:
            MCC = 0
        else:
            MCC = (tp * tn - fp * fn) / (np.sqrt(tp + fn) * np.sqrt(tp + fp) * np.sqrt(tn + fp) * np.sqrt(tn + fn))

        if tp == 0:
            Recall = 0
            Precision = 0
        else:
            Recall = tp / (tp + fn)
            Precision = tp / (tp + fp)

        res_mcc.append(round(MCC, 3))
        res_acc.append(round(ACC, 3))
        res_auc.append(round(AUC, 3))
        res_aupr.append(round(AUPR, 3))
        res_recall.append(round(Recall, 3))
        res_precision.append(round(Precision, 3))
        res_f1.append(round(F1, 3))

    df = pd.DataFrame({'ACC': res_acc, 'AUC': res_auc, 'MCC': res_mcc, 'AUPR': res_aupr, 'F1': res_f1,
                       'Precision': res_precision, 'Recall': res_recall}, index=class_names)
    if show:
        print(df)

    if save is not None:
        df.to_csv(save)

def weighted_overall_metrics(y_pred: np.array, y_true: np.array, class_names, thresholds, save = None, show = True):
    """
    综合评价多标签分类任务, 每个类别一个阈值
    """
    y_pred_cls = np.zeros_like(y_pred)
    for i, pt in enumerate(class_names):
        y_pred_cls[:, i][y_pred[:, i] >= thresholds[pt]] = 1    # 预测类别


    HLoss = hamming_loss(y_true, y_pred_cls)

    # Calculate metrics globally by counting the total true positives,false negatives and false positives.
    F1_micro = f1_score(y_true, y_pred_cls, average='micro')

    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    F1_macro = f1_score(y_true, y_pred_cls, average='macro')

    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
    # This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.

    F1_weighted = f1_score(y_true, y_pred_cls, average='weighted')

    df = pd.DataFrame({'HLoss': [HLoss], 'F1_micro': [F1_micro], 'F1_macro': [F1_macro], 'F1_weighted': [F1_weighted]})
    if show:
        print(df)

    if save is not None:
        df.to_csv(save)

    return df

def failed_predicted_instance(y_pred: np.array, y_true: np.array, class_names, seqs, threshold=0.5, save = None):
    """
    保存每个类别中预测错误的序列, FP, FN, 置信度
    FP, FN分别按照置信度(正样本分数)从大到校小排序
    """
    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred >= threshold] = 1    # 预测类别

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        err = []
        err_type = []
        err_sup = []

        df = pd.DataFrame(columns=['Seq', 'Type', 'Support'])

        for i in range(len(y_c)):
            if y_t[i] == pos and y_c[i] == neg:
                err.append(seqs[i])
                err_type.append('FN')
                err_sup.append(y_p[i])

            elif y_t[i] == neg and y_c[i] == pos:
                err.append(seqs[i])
                err_type.append('FP')
                err_sup.append(y_p[i])

        df['Seq'] = err
        df['Type'] = err_type
        df['Support'] = err_sup

        if save is not None:
            df.to_csv(os.path.join(save, class_names[c] + '.csv'), index=False)