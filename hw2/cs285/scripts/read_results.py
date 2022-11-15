import glob
import numpy as np

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib.pyplot as plt


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y


def get_logdir_results_into_df(logdir):
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile)
    return pd.DataFrame({
        'iter': np.arange(len(X)),
        'train_steps': X,
        'return': Y
    })


if __name__ == '__main__':

    logdirs = [
        'data/q1_lb_no_rtg_dsa_CartPole-v0_05-11-2022_08-38-32/events*',
        'data/q1_lb_rtg_dsa_CartPole-v0_08-11-2022_09-41-11/events*',
        'data/q1_lb_rtg_na_CartPole-v0_08-11-2022_09-44-06/events*'
    ]
    for logdir in logdirs:
        df = get_logdir_results_into_df(logdir)
        plt.plot(df['train_steps'], df['return'], label=logdir)
    plt.legend()
    plt.show()