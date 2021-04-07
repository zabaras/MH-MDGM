
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import os
plt.switch_backend('agg')


class plot_error_bar(object):
    def __init__(self,device,save_dir,file_name):
        self.device = device
        self.save_dir = save_dir
        self.file_name = file_name

    def errorfill(self,x, y, yerr, color=None, alpha_fill=0.3, ax=None):
        ax = ax if ax is not None else plt.gca()
        if color is None:
            color = ax._get_lines.color_cycle.next()
        if np.isscalar(yerr) or len(yerr) == len(y):
            ymin = y - yerr
            ymax = y + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr
        ax.plot(x, y, color=color)
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

    def error_bar(self,actual,pred):
        '''
        Args: 
        actual is ground truth with dimension: 64X64
        pred is the prediction samples with dimension: samplesX64X64
        '''
        actual = np.log(actual).reshape(64,64)
        pred = pred.reshape(-1,64,64)
        pred_mean = np.mean(pred,axis=0)
        pred_mean = pred_mean.reshape(64,64)
        pred_diag = np.diag(pred_mean)

        actdiag = np.diag(actual)

        pred_std = np.std(pred,axis=0)
        pred_std = pred_std.reshape(64,64)
        std_diag = np.diag(pred_std)

        std_val = np.std(actdiag)
        x = np.linspace(0, 64,64)
        y_sin = np.sin(x)
        y_cos = np.cos(x)
        self.errorfill(x, pred_diag, 2*std_diag, 'red')
        plt.plot(x,actdiag,'black')
        plt.ylim(-0.5,3)

        plt.savefig(self.save_dir+self.file_name, bbox_inches='tight')
        plt.close()

