import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from sklearn import metrics as skmetrics
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm


def auroc(labels, probs):
    fpr, tpr, thresholds = skmetrics.roc_curve(labels, probs, pos_label=1)
    return {
        "auroc": skmetrics.auc(fpr, tpr),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def auprc(labels, probs):
    precision, recall, thresholds = skmetrics.precision_recall_curve(labels, probs)
    return {
        "auprc": skmetrics.auc(recall, precision),
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }

def accuracy(probs, labels, n_thresholds=200):
    thresholds = tf.cast(tf.linspace(1, 0, n_thresholds), probs.dtype)
    decisions = probs[jnp.newaxis, :] - thresholds[:, jnp.newaxis] > 0
    labels = jnp.squeeze(labels)
    oneone = (
        tf.cast(labels == 1, dtype=probs.dtype) *
        tf.cast(decisions, dtype=probs.dtype))

    onezero = (
        tf.cast(labels == 1, dtype=probs.dtype) *
        (1-tf.cast(decisions, dtype=probs.dtype))
    )
    TP = tf.reduce_sum(oneone, axis=-1)
    FN = tf.reduce_sum(onezero, axis=-1)
    TPR = TP/(TP+FN)
    # TPR = tf.pad(TPR, [(0, 0), (1, 0)], "CONSTANT")

    zeroone = (
        tf.cast(labels == 0, dtype=probs.dtype) *
        tf.cast(decisions, dtype=probs.dtype))
    zerozero = (
        tf.cast(labels == 0, dtype=probs.dtype) *
        (1-tf.cast(decisions, dtype=probs.dtype)))

    FP = tf.reduce_sum(zeroone, axis=-1)
    TN = tf.reduce_sum(zerozero, axis=-1)
    FPR = FP/(FP+TN)
    # FPR = tf.pad(FPR, [(0, 0), (1, 0)], "CONSTANT")
    precision = TP/(TP+FP)
    precision = tf.where(
        tf.math.is_finite(precision),
        precision,
        jnp.zeros_like(precision)
    )
    recall = TP/(TP+FN)
    auroc = auc(FPR, TPR)
    auprc = auc(recall, precision)
    return {
        'tpr': TPR, 'fpr': FPR, 'precision': precision, 'recall': recall,
        'auroc': auroc, 'auprc': auprc
        }


def auc(x, y):
    return tfp.math.trapz(x=x, y=y)

def classification_metrics(
    data_factory,
    prediction_fn,
    preprocessing_fn=None,
    by_vars=None,
    outcome_label="label",
    method="pd",
    save_file=None,
):
    if by_vars is None:
        by_vars = []
    collect_vars = by_vars + [outcome_label]
    collect_vars = set(collect_vars)
    collected_data = {k: [] for k in collect_vars}
    probs = []
    metrics = {}

    for batch in tqdm(iter(data_factory())):
        if preprocessing_fn is not None:
            batch = preprocessing_fn(batch)

        for k in collected_data.keys():
            collected_data[k] += [jnp.squeeze(batch[k])]

        probs += [prediction_fn(data=batch)]

    probs = np.concatenate(probs, axis=0)
    for k in collected_data.keys():
        collected_data[k] = np.squeeze(np.concatenate([np.array(x).flatten() for x in collected_data[k]], axis=0))

    computed = pd.DataFrame({"probs": probs, **collected_data})
    if save_file:
        computed.to_parquet(save_file)
    metrics["prob"] = np.mean(computed[outcome_label])
    metrics["auroc"] = auroc(computed[outcome_label], computed.probs)
    metrics["auprc"] = auprc(computed[outcome_label], computed.probs)

    for var in by_vars:
        metrics[var] = {}
        metrics[var]["prob"] = (
            computed.groupby(var)
            .apply(lambda x: np.mean(x[outcome_label]))
            .dropna()
            .reset_index()
        )
        metrics[var]["count"] = (
            computed.groupby(var)
            .apply(lambda x: np.sum(x[outcome_label]))
            .dropna()
            .reset_index()
        )
        metrics[var]["auroc"] = (
            computed.groupby(var)
            .apply(lambda x: auroc(x[outcome_label], x.probs))
            .dropna()
            .reset_index()
        )
        metrics[var]["auroc"] = pd.concat(
            [metrics[var]["auroc"], metrics[var]["auroc"][0].apply(pd.Series)], axis=1
        ).drop(columns=0)
        metrics[var]["auprc"] = (
            computed.groupby(var)
            .apply(lambda x: auprc(x[outcome_label], x.probs))
            .dropna()
            .reset_index()
        )
        metrics[var]["auprc"] = pd.concat(
            [metrics[var]["auprc"], metrics[var]["auprc"][0].apply(pd.Series)], axis=1
        ).drop(columns=0)
    return metrics


