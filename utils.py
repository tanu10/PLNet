"""General utility functions"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import json
import logging
import csv
import scipy.io as io
import torch
import numpy as np

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


# def row_csv2dict(csv_file):
#     dict_club={}
#     with open(csv_file)as f:
#         reader=csv.reader(f,delimiter=',')
#         for row in reader:
#             dict_club[(row[0],row[1])]=row[2]
#     return dict_club


def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'model.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    # else:
    #     print("Checkpoint Directory exists! ")
    torch.save(state, filepath)



def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['gen_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def plot_histogram(Effs, Iter, fig_path):
    ax = plt.figure()
    bins = [i*5 for i in range(21)]
    plt.hist(Effs*100, bins, facecolor='blue', alpha=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.yticks([])
    plt.xticks(fontsize=12)
    #plt.yticks(fontsize=20)
    plt.xlabel('Deflection efficiency (%)', fontsize=12)
    plt.title('Iteration {}'.format(Iter), fontsize=16)
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_spectrum(spectrum, bm, fn):
    x = [i for i in range(380, 781)]
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(color="lightgrey")
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("PPFD ($\mu mol/ms^2$)", fontsize=12)
    plt.title("PPFD = "+"{:.2f}".format(spectrum.sum())+" Biomass = " + "{:.2f}".format(bm))
    rainbow_fill(x, spectrum)
    #ax.set_ylim(bottom=0)
    plt.show()
    plt.savefig(fn)
    plt.close()


def polygon(x1, y1, x2, y2, c):
    ax = plt.gca()
    polygon = plt.Polygon([(x1, y1), (x2, y2), (x2, 0), (x1, 0)], color=c)
    ax.add_patch(polygon)


def rainbow_fill(X, Y, cmap=plt.get_cmap("jet")):
    plt.plot(X, Y, lw=0)  # Plot so the axes scale correctly

    dx = X[1]-X[0]
    N = float(len(X)) - 80

    for n, (x, y) in enumerate(zip(X[:-80], Y[:-80])):
        color = cmap(n/N)
        if n+1 == N:
            continue
        polygon(x, y, X[n+1], Y[n+1], color)
    N = 80
    for _, (x, y) in enumerate(zip(X[-80:], Y[-80:])):
        if n+1 == N:
            continue
        polygon(x, y, X[n+1], Y[n+1], color)
        n = n+1


