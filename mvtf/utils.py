import pickle
import os
import pathlib
import logging
import numpy as np
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

strBold = lambda skk: "\033[1m {}\033[00m".format(skk)
strBlue = lambda skk: "\033[34m {}\033[00m".format(skk)
strRed = lambda skk: "\033[91m {}\033[00m".format(skk)
strGreen = lambda skk: "\033[92m {}\033[00m".format(skk)
strYellow = lambda skk: "\033[93m {}\033[00m".format(skk)
strLightPurple = lambda skk: "\033[94m {}\033[00m".format(skk)
strPurple = lambda skk: "\033[95m {}\033[00m".format(skk)
strCyan = lambda skk: "\033[96m {}\033[00m".format(skk)
strLightGray = lambda skk: "\033[97m {}\033[00m".format(skk)
strBlack = lambda skk: "\033[98m {}\033[00m".format(skk)

prBold = lambda skk: print("\033[1m {}\033[00m".format(skk))
prBlue = lambda skk: print("\033[34m {}\033[00m".format(skk))
prRed = lambda skk: print("\033[91m {}\033[00m".format(skk))
prGreen = lambda skk: print("\033[92m {}\033[00m".format(skk))
prYellow = lambda skk: print("\033[93m {}\033[00m".format(skk))
prLightPurple = lambda skk: print("\033[94m {}\033[00m".format(skk))
prPurple = lambda skk: print("\033[95m {}\033[00m".format(skk))
prCyan = lambda skk: print("\033[96m {}\033[00m".format(skk))
prLightGray = lambda skk: print("\033[97m {}\033[00m".format(skk))
prBlack = lambda skk: print("\033[98m {}\033[00m".format(skk))


def setup_logging(log_dir):
    # log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : " \
    #                   "%(message)s in %(pathname)s:%(lineno)d"
    log_file_format = "[%(asctime)s: %(message)s"
    log_console_format = "%(message)s"

    # Main logger
    main_logger = logging.getLogger()
    if main_logger.hasHandlers():
        for handler in main_logger.handlers:
            main_logger.removeHandler(handler)

    main_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir),
                                           maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir),
                                                  maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.ERROR)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def sigmoid(x, derivative=False):
    """
    compute the sigmoid function 1 / (1 + exp(-x))
    :param x: input of sigmoid function
    :param derivative: boolean value, if True compute the derivative of sigmoid function instead
    :return:
    """
    if x > 100:
        sigm = 1.
    elif x < -100:
        sigm = 0.
    else:
        sigm = 1. / (1. + np.exp(-x))

    if derivative:
        return sigm * (1. - sigm)
    return sigm
