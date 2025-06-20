import numpy as np
import os
import matplotlib.pyplot as plt


def area_greater_than(
    data, variable, threshs=[0, 1, 3, 5, 10, 15, 20, 25, 30]#, 40, 50]
):
    """
    Extract from each grid all the values greater than threshold and compute their area proportion

    Args:
        variable (int): index of the channels corresponding to the variable
        data (array): array of the loaded data, shape N x C x H x W
        thresholds (list):me    list of thresholds
    Returns:
        np.array[float]: store every value greater than the threshold
    """
    print("je passe par la ")
    mean_proportion = np.zeros((len(threshs),))
    for idx_threshold, threshold in enumerate(threshs):
        # mask = np.exp((data[:, variable]+1)*5.78319931/2)-1 > threshold
        # extracted = (np.exp((data[:, variable]+1)*5.78319931/2)-1)[mask]
        print('je suis la shape des data',data.shape,variable)
        mask = data[:, variable] > threshold
        extracted = data[:, variable][mask]
        mean_proportion[idx_threshold] += len(extracted)

    mean_proportion /= data.shape[-2] * data.shape[-1] * data.shape[0]

    return mean_proportion



