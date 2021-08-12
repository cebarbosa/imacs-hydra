import platform

import matplotlib.pyplot as plt

node = platform.node()
lai_machines = ["uv100", "alphacrucis", "yaci.iag.usp.br"] # nodes @LAI cluster
# Settings for personal computer
if node in ["kadu-Inspiron-5557"]:
    home_dir = "/home/kadu/Dropbox/imacs-imf"
    cvd_dir = "/home/kadu/Dropbox/SSPs/CvD18"
    mp_pool_size = 4 # Number of cores for parallel processing
# Settings for supercomputers @IAG/USP
elif node in lai_machines:
    home_dir = "/sto/home/cebarbosa/imacs-imf"
    cvd_dir = "/sto/home/cebarbosa/SSPs/CvD18"
    mp_pool_size = 64
else:
    raise ValueError("Please define your directories inside context.py")

# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set tick width
width = 0.5
majsize = 4
minsize = 2
plt.rcParams['xtick.major.size'] = majsize
plt.rcParams['xtick.major.width'] = width
plt.rcParams['xtick.minor.size'] = minsize
plt.rcParams['xtick.minor.width'] = width
plt.rcParams['ytick.major.size'] = majsize
plt.rcParams['ytick.major.width'] = width
plt.rcParams['ytick.minor.size'] = minsize
plt.rcParams['ytick.minor.width'] = width
plt.rcParams['axes.linewidth'] = width

fig_width = 3.54 # inches - A&A template