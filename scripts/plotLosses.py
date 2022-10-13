import os
import jaxopt
import numpy as np
import pandas as pd
import tinygp
import jax
import jax.numpy as jnp
from io import StringIO
import matplotlib.pyplot as plt
from plotting import *
from helper_functions import *
import pickle
from jax.config import config
import glob

inDir = "/Users/alexgagliano/Documents/Research/HostClassifier/packages/phast/"
inFile = "Model_Output.txt"

#plot training and validation loss and accuracy
loss, val_loss = plotTrainingHistory(inDir, inFile)
