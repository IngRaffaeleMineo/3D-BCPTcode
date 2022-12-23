import numpy as np
import io
from sklearn import metrics
import PIL.Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import itertools
from sklearn import metrics
from mpl_toolkits.axes_grid1 import ImageGrid
import math
from tqdm import tqdm


############
## GRAPHs ##


def plot_confusion_matrix(tot_true_labels, tot_predicted_labels, classes, title='Confusion matrix', normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = metrics.confusion_matrix(tot_true_labels, tot_predicted_labels)

    plt.ioff()
    fig, ax = plt.subplots()
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(img, ax=ax)
    ax.title.set_text(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks) 
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks) 
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_graph(x, y, xlabel, ylabel):
    """
    This function plots a curve and export it in image.
    """
    plt.ioff()
    fig, ax = plt.subplots()

    ax.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_histogram(hist, bins, x_tick_vertical=False, x_label="Value", y_label="Count", title='Histogram'):
    """
    This function prints and plots a histogram.
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ax.hist(hist, bins=bins, alpha=0.5)

    if x_tick_vertical:
        plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_double_histogram(hist1, hist2, labels, bins, x_tick_vertical=False, x_label="Value", y_label="Count", title='Double Histogram'):
    """
    This function prints and plots two histograms on same graph.
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ax.hist(hist1, bins=bins[0], alpha=0.5, label=labels[0])
    ax.hist(hist2, bins=bins[1], alpha=0.5, label=labels[1])

    if x_tick_vertical:
        plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper right')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_bar(labels, counts, x_tick_vertical=False, x_label="Value", y_label="Count", title='Bar plot'):

    """
    This function prints and plots a bar-plot.
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ticks = range(len(counts))
    plt.bar(ticks,counts, align='center')
    plt.xticks(ticks, labels, rotation='vertical')

    if x_tick_vertical:
        plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)
    
    return image


def plotImages(title, image):
    # image.shape = (CHANNEL,M,N)
    _,ax = plt.subplots()
    ax.title.set_text(title)
    ax.imshow(image.permute(1,2,0))


def saveGridImages(filename,imgs,n_colonne=10, figsize=100):
    plt.ioff()
    fig, axes = plt.subplots(nrows=math.ceil(len(imgs)/n_colonne), ncols=n_colonne, figsize=(figsize,figsize))
    for idx, image in enumerate(imgs):
        row = idx // n_colonne
        col = idx % n_colonne
        axes[row, col].axis("off")
        axes[row, col].set_title(idx)
        axes[row, col].imshow(image, cmap="gray", aspect="equal")
    for idx in range(len(imgs),math.ceil(len(imgs)/n_colonne)*n_colonne):
        row = idx // n_colonne
        col = idx % n_colonne
        axes[row, col].axis("off")
    plt.subplots_adjust(wspace=.0, hspace=.0)
    plt.savefig(filename + ".jpg")
    plt.clf()
    plt.close(fig)
    del fig


#############
## METRICs ##


def calc_tp_fp_tn_fn(correct_labels_in,predicted_labels_in):
    correct_labels, predicted_labels = np.array(correct_labels_in), np.array(predicted_labels_in)

    TP = (((correct_labels==1).astype(int) + (predicted_labels==1).astype(int)) == 2).sum().item()
    FP = (((correct_labels==0).astype(int) + (predicted_labels==1).astype(int)) == 2).sum().item()
    TN = (((correct_labels==0).astype(int) + (predicted_labels==0).astype(int)) == 2).sum().item()
    FN = (((correct_labels==1).astype(int) + (predicted_labels==0).astype(int)) == 2).sum().item()

    return TP,FP,TN,FN


def calc_precision(correct_labels,predicted_labels):
    TP, FP, _, _ = calc_tp_fp_tn_fn(correct_labels, predicted_labels)

    try:
        precision = TP/(TP+FP)
    except ZeroDivisionError:
        precision = 0.0
    
    return precision


def calc_recall(correct_labels,predicted_labels):
    TP, _, _, FN = calc_tp_fp_tn_fn(correct_labels, predicted_labels)

    try:
        recall = TP/(TP+FN)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def calc_specificity(correct_labels,predicted_labels):
    _, FP, TN, _ = calc_tp_fp_tn_fn(correct_labels, predicted_labels)

    try:
        specificity = TN/(TN+FP)
    except ZeroDivisionError:
        specificity = 0.0
    
    return specificity


def calc_f1(correct_labels,predicted_labels):
    precision = calc_precision(correct_labels, predicted_labels)
    recall = calc_recall(correct_labels, predicted_labels)

    try:
        f1score = 2*(precision*recall) / (precision+recall)
    except ZeroDivisionError:
        f1score = 0.0
    
    return f1score


def calc_accuracy_classification(correct_labels,predicted_labels):
    TP, FP, TN, FN = calc_tp_fp_tn_fn(correct_labels, predicted_labels)

    correct = TP + TN
    total = TP + FP + TN + FN
    accuracy = correct / total

    return accuracy


def calc_accuracy_balanced_classification(correct_labels,predicted_labels):
    recall = calc_recall(correct_labels, predicted_labels)
    specificity = calc_specificity(correct_labels, predicted_labels)
    
    accuracy_balanced = (recall+specificity) / 2

    return accuracy_balanced


def calc_accuracy_regression(correct_labels_in,predicted_labels_in, soglia):
    correct_labels, predicted_labels = np.array(correct_labels_in), np.array(predicted_labels_in)

    labels_finite = correct_labels[np.isfinite(correct_labels)]
    predicted_finite = predicted_labels[np.isfinite(correct_labels)]

    correct = (np.abs(labels_finite-predicted_finite) <= soglia).sum()
    total = labels_finite.shape[0]

    try:
        accuracy_regression = correct / total
    except ZeroDivisionError:
        accuracy_regression = 0.0

    return accuracy_regression


def calc_accuracy_regression_labels(correct_labels, predicted_regression_in, soglia):
    predicted_regression = np.array(predicted_regression_in)
    predicted_labels = (predicted_regression<=soglia).astype(int)
    
    accuracy_regression_labels = calc_accuracy_classification(correct_labels,predicted_labels)

    return accuracy_regression_labels


def calc_accuracyBalanced_regression_labels(correct_labels, predicted_regression_in, soglia):
    predicted_regression = np.array(predicted_regression_in)
    predicted_labels = (predicted_regression<=soglia).astype(int)
    
    accuracy_balanced_regression_labels = calc_accuracy_balanced_classification(correct_labels,predicted_labels)

    return accuracy_balanced_regression_labels


def calc_auc(correct_labels_in,predicted_scores_in):
    correct_labels, predicted_scores = np.array(correct_labels_in), np.array(predicted_scores_in)

    mask_nan_inf = np.all(np.isfinite(predicted_scores), axis=1)
    correct_labels, predicted_scores = correct_labels[mask_nan_inf], predicted_scores[mask_nan_inf]

    roc_auc = metrics.roc_auc_score(correct_labels,predicted_scores[:,1])
    
    return roc_auc


def calc_rocCurve(correct_labels_in,predicted_scores_in):
    correct_labels, predicted_scores = np.array(correct_labels_in), np.array(predicted_scores_in)

    mask_nan_inf = np.all(np.isfinite(predicted_scores), axis=1)
    correct_labels, predicted_scores = correct_labels[mask_nan_inf], predicted_scores[mask_nan_inf]

    fpr, tpr, _ = metrics.roc_curve(correct_labels,predicted_scores[:,1])

    rocCurve_image = plot_graph(fpr,tpr, "False Positive Rate", "True Positive Rate")
    
    return rocCurve_image


def calc_precisionRecallCurve(correct_labels_in,predicted_scores_in):
    correct_labels, predicted_scores = np.array(correct_labels_in), np.array(predicted_scores_in)

    mask_nan_inf = np.all(np.isfinite(predicted_scores), axis=1)
    correct_labels, predicted_scores = correct_labels[mask_nan_inf], predicted_scores[mask_nan_inf]

    precision, recall, _ = metrics.precision_recall_curve(correct_labels,predicted_scores[:,1])

    precisionRecallCurve_image = plot_graph(recall,precision, "Recall", "Precision")
    
    return precisionRecallCurve_image


###################
## OTHER METRICs ##


def calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last, tot_image_paths):
    tot_predicted_labels_last_split = {tot_image_paths[i]:tot_predicted_labels[i] for i in range(len(tot_image_paths))}

    predictionAgreement=0
    predictionAgreement_tot=0
    for key in tot_predicted_labels_last_split:
        if key not in tot_predicted_labels_last.keys():
            continue
        predictionAgreement_tot += 1

        if tot_predicted_labels_last_split[key] == tot_predicted_labels_last[key]:
            predictionAgreement += 1
    
    try:
        predictionAgreementRate = predictionAgreement/predictionAgreement_tot
    except ZeroDivisionError:
        predictionAgreementRate = 0.0

    return predictionAgreementRate, tot_predicted_labels_last_split


def calc_FN_FP_histograms(tot_true_labels, tot_predicted_labels, tot, name, split):
    correct_labels, predicted_labels = np.array(tot_true_labels), np.array(tot_predicted_labels)

    FP_index = (((correct_labels==0).astype(int) + (predicted_labels==1).astype(int)) == 2)
    FN_index = (((correct_labels==1).astype(int) + (predicted_labels==0).astype(int)) == 2)

    tot_FP = np.array(tot)[FP_index]
    tot_FN = np.array(tot)[FN_index]

    tot_FP = tot_FP[np.isfinite(tot_FP)]
    tot_FN = tot_FN[np.isfinite(tot_FN)]

    hist_image = plot_double_histogram(tot_FP, tot_FN, ["FP","FN"], [100, 100], x_tick_vertical=False, x_label=(name + " value"), y_label="Count", title="Histograms " + name + " - "+split)

    return hist_image


def calc_predictionError_histograms(labels_in, predicted_in, soglia, name, split):
    labels, predicted = np.array(labels_in), np.array(predicted_in)
    
    labels_finite = labels[np.isfinite(labels)]
    predicted_finite = predicted[np.isfinite(labels)]

    wrong = np.abs(np.subtract(labels_finite,predicted_finite)) > soglia

    predictionError = labels_finite[wrong]

    hist_image = plot_histogram(predictionError, bins=100, x_tick_vertical=False, x_label=(name + " value"), y_label="Count", title="Histogram error prediction " + name + " - "+split)

    return hist_image

