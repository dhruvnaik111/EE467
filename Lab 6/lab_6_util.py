import time
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, confusion_matrix

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

def evaluate_model(model, name, feat_test, y_test):
    """ Evaluate a classification model on the test set, then print and plot metrics. """
    # Make prediction from features
    pred_test = model.predict(feat_test)
    
    print(f"[ Evaluation result for {name} ]")
    # Print classification report
    print("Classification report:")
    print(classification_report(y_test, pred_test))
    
    # Print confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred_test), "\n")