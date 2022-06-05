import tensorflow as tf


def accuracy(probs, labels, n_thresholds=200):
    thresholds = tf.cast(tf.linspace(0, 1, n_thresholds), probs.dtype)
    decisions = probs[tf.newaxis, :] - thresholds[:, tf.newaxis] > 0
    oneone = (
        tf.cast(labels == 1, dtype=probs.dtype) *
        tf.cast(decisions, dtype=probs.dtype))

    onezero = (
        tf.cast(labels == 1, dtype=probs.dtype) *
        (1-tf.cast(decisions, dtypep=probs.dtype))
    )
    TP = tf.reduce_sum(oneone, axis=-1)
    FN = tf.reduce_sum(onezero, axis=-1)
    TPR = TP/(TP+FN)
    TPR = tf.pad(TPR, [(0, 0), (1, 0)], "CONSTANT")

    zeroone = (
        tf.cast(labels == 0, dtype=probs.dtype) *
        tf.cast(decisions, dtype=probs.dtype))
    zerozero = (
        tf.cast(labels == 0, dtype=probs.dtype) *
        (1-tf.cast(decisions, dtype=probs.dtype)))

    FP = tf.reduce_sum(zeroone, axis=-1)
    TN = tf.reduce_sum(zerozero, axis=-1)
    FPR = FP/(FP+TN)
    FPR = tf.pad(FPR, [(0, 0), (1, 0)], "CONSTANT")
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return {'tpr': TPR, 'fpr': FPR, 'precision': precision, 'recall': recall}


def auc(x, y):
    pass
