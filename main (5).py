import pandas as pd
import numpy as np
import Data_treatment
from Data_treatment import *


def preprocess(train_set, test_set):
    data_normalization_train(train_set)
    main_preprocessing_test(test_set)
