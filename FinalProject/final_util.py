import time
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

@contextmanager
def timeit(action="Timing"):
    """ Print the execution time of certain Python operations. """
    # Record start time
    print(f"{action} started...")
    start_time = time.time()
    
    # Execute task
    yield
    
    # Compute and show elapsed time
    elapsed_time = time.time()-start_time
    print(f"{action} completed. Elapsed time: {elapsed_time:.2f}s\n")

# Name: recall_at_fixed_fpr
# Description: Calculate recall at a specific false positive rate (FPR) threshold. (Defaults to 1% FPR)
def recall_at_fixed_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the indices where FPR is less than or equal to our target (e.g., 1%)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    # Return the recall at the highest allowed FPR
    return tpr[idx[-1]]

# Name: select_threshold_at_fpr
# Description: Find the probability threshold that corresponds to a specific FPR (e.g., 1% FPR) on the validation set.
def select_threshold_at_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.5
    return thresholds[idx[-1]]