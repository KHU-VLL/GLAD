import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def split_cluster_acc_v2(y_true, y_pred, mask, return_conf=False, return_indmap=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}  # pred -> gt
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for l in old_classes_gt:
        old_acc += w[ind_map[l], l]  # shuffled pred, gt
        total_old_instances += sum(w[:, l])
    old_acc /= total_old_instances  # # correct old / # all old

    new_acc = 0
    total_new_instances = 0
    for l in new_classes_gt:
        new_acc += w[ind_map[l], l]  # shuffled pred, gt
        total_new_instances += sum(w[:, l])
    if total_new_instances != 0:  # if there is at least one new instances
        new_acc /= total_new_instances

    result = [total_acc, old_acc, new_acc]
    if return_conf:
        # confmat `w` in this function: row-pred, col-gt
        # confmat to be: row-gt, col-pred
        ind_reverse = np.argsort(ind[:,-1])
        result += [w[ind_reverse].T]
    if return_indmap:
        result += [ind[:,-1]]
    return result


def split_cluster_acc_v2_balanced(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean() if new_acc.size != 0 else 0
    return total_acc, old_acc, new_acc


if __name__ == '__main__':
    import re
    import inspect

    END = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'

    def pprint_conf(conf, num_old_classes):
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            s = str(conf)
        w, h = conf.shape
        for (ii, jj), (start, end) in reversed([((i//w, i%h), (m.start(0), m.end(0))) for i, m in enumerate(re.finditer(r'\d+', str(s))) if i%h == i//w]):
            s = s[:start] + BOLD + (GREEN if max(ii, jj) < num_old_classes else PURPLE) + s[start:end] + END + s[end:]  # diag vals to be bold
        print(s)

    def test1():
        print('Normal Case\n')
        num_all_classes = 30  # example
        num_old_classes = 16  # example
        gt = np.random.randint(num_all_classes, size=100*num_all_classes)
        pred = gt.copy()
        wrong_mask = np.random.choice(gt.shape[0], gt.shape[0]//2, replace=False)
        pred[wrong_mask] = np.random.randint(num_all_classes, size=wrong_mask.shape[0])
        old_mask = gt < num_old_classes
        total_acc, old_acc, new_acc, conf = split_cluster_acc_v2(gt, pred, old_mask, True)
        print(inspect.cleandoc(f'''
            Fake K-means:
                ALL: {total_acc:.4f}
                Old: {old_acc:.4f}
                New: {new_acc:.4f}
        '''))
        pprint_conf(conf, num_old_classes)
        print('\n\n')

    def test2():
        print('No New Samples\n')
        num_all_classes = 20
        num_old_classes = 10
        gt = np.random.randint(num_old_classes, size=100*num_all_classes)  # to check if 0-division handled
        is_closed_set = gt.max()+1 <= num_old_classes
        if is_closed_set:
            num_all_classes = num_old_classes
        pred = gt.copy()
        wrong_mask = np.random.choice(gt.shape[0], gt.shape[0]//2, replace=False)
        pred[wrong_mask] = np.random.randint(num_all_classes, size=wrong_mask.shape[0])
        old_mask = gt < num_old_classes
        total_acc, old_acc, new_acc, conf = split_cluster_acc_v2(gt, pred, old_mask, True)
        print(inspect.cleandoc(f'''
            Fake K-means:
                ALL: {total_acc:.4f}
                Old: {old_acc:.4f}
                New: {new_acc:.4f}
        '''))
        pprint_conf(conf, num_old_classes)
        print('\n\n')

    test1()
    test2()
