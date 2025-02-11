from copy import deepcopy

import numpy as np


def Intercor_Member_Matrix(real_data, fake_data):
    """
    compute the Intercorrelation between member

    Inputs :
        real_data, fake_data : numpy arrays, shape B x C x N x N
        with C the different channels where spectrograms are independently
        computed

    Returns :

        res : numpy array, shape C (output array)
    """
    num_real_member, c_real, _, _ = real_data.shape
    num_fake_member, c_fake, _, _ = fake_data.shape

    return None
