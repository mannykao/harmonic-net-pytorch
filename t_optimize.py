"""
Title: T Optimize

Desc: Rough File used to optimize various modules of this repo.

Authors: Ujjawal K. Panchal and Manny Ko
"""
#generic imports.
import pickle, torch, numpy as np

from scipy.linalg import dft
from typing import Union, Callable
from pathlib import Path

#files of the repo.
import hnet_ops



#optimized implementation of functions.
def get_interpolation_weights(fs, m, n_rings=None):
    '''
    Used to construct the steerable filters using Radial basis functions.
    The filters are constructed on the patches of n_rings using Gaussian
    interpolation. (Code adapted from the tf code of Worrall et al, CVPR, 2017)

    Args:
        fs (int): filter size for the H-net convoutional layer
        m (int): max. rotation order for the steerbable filters
        n_rings (int): No. of rings for the steerbale filters

    Returns:
        norm_weights (numpy): contains normalized weights for interpolation
        using the steerable filters
    '''

    if n_rings is None:
        n_rings = np.maximum(fs/2, 2)

    # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
    radii = np.linspace(m!=0, n_rings-0.5, n_rings)

    # We define pixel centers to be at positions 0.5
    center_pt = np.asarray([fs, fs])/2.

    # Extracting the set of angles to be sampled
    N = hnet_ops.get_sample_count(fs)

    # Choosing the sampling locations for the rings
    lin = (2*np.pi*np.arange(N))/N
    ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

    # Create interpolation coefficient coordinates
    coords = hnet_ops.get_l2_neighbors(center_pt, fs)

    # getting samples based on the choisen center_pt and the coords
    radii = radii[:,np.newaxis,np.newaxis,np.newaxis]
    ring_locations = ring_locations[np.newaxis,:,:,np.newaxis]
    diff = radii*ring_locations - coords[np.newaxis,:,np.newaxis,:]
    dist2 = np.sum(diff**2, axis=1)

    # Convert distances to weightings
    weights = np.exp(-0.5*dist2/(0.5**2)) # For bandwidth of 0.5

    # Normalizing the weights to calibrate the different steerable filters
    norm_weights = weights/np.sum(weights, axis=2, keepdims=True)
    return norm_weights


def get_filter_weights(R_dict, fs, P=None, n_rings=None):
    '''
    Calculates filters in the form of weight matrices through performing
    single-frequency DFT on every ring obtained from sampling in the polar 
    domain. 

    Args:
        R_dict (dict): contains initialization weights
        fs (int): filter size for the h-net convolutional layer

    Returns:
        W (dict): contains the filter matrices
    '''
       
    k = fs
    W = {} # dict to store the filter matrices
    N = hnet_ops.get_sample_count(k)

    for m, r in R_dict.items():
        rsh = list(r.size())

        # Get the basis matrices built from the steerable filters
        weights = hnet_ops.get_interpolation_weights(k, m, n_rings=n_rings)
        DFT = dft(N)[m,:]
        low_pass_filter = np.dot(DFT, weights).T

        cos_comp = np.real(low_pass_filter).astype(np.float32)
        sin_comp = np.imag(low_pass_filter).astype(np.float32)

        # Arranging the two components in a manner that they can be directly
        #  multiplied with the steerable weights
        cos_comp = torch.from_numpy(cos_comp)
        cos_comp = cos_comp.to(device="cuda" if torch.cuda.is_available() else "cpu")
        sin_comp = torch.from_numpy(sin_comp)
        sin_comp = sin_comp.to(device="cuda" if torch.cuda.is_available() else "cpu")

        # Computng the projetions on the rotational basis
        r = r.view(rsh[0],rsh[1]*rsh[2])
        ucos = torch.matmul(cos_comp, r).view(k, k, rsh[1], rsh[2]).double()
        usin = torch.matmul(sin_comp, r).view(k, k, rsh[1], rsh[2]).double()

        if P is not None:
            # Rotating the basis matrices
            ucos_ = torch.cos(P[m])*ucos + torch.sin(P[m])*usin
            usin = -torch.sin(P[m])*ucos + torch.cos(P[m])*usin
            ucos = ucos_
        W[m] = (ucos, usin)

    return W


#test functions.
def compare_filter_weight_output(d1: dict, d2: dict, tol = 1e-09):
    #1. if keys not same.
    if d1.keys() != d2.keys():
        return False

    #2. check values.
    for key in d1.keys():
        d1_stack = torch.vstack(d1[key])
        d2_stack = torch.vstack(d2[key])
        if not torch.allclose(d1_stack, d2_stack, atol = tol):
            return False

    #3. if all keys are equal, the output is equal.
    return True



def test_function(
    dictfile: Union[Path, str],
    new_func: Callable,
    eq_test: Callable = lambda x, y: x == y
) -> bool:
    """
    Desc:
        Function to test that both old and the new implementation of the function produce the same result.

    Args:
        1. dictfile:
            A pickle dump of a Tuple(input_dict, output_dict).
            Where: `input_dict` has input params to func, `output_dict` has output obtained from old function.
        2. new_func: New implementation of the old function. 
    """
    #1. load .iodict file.
    with open(dictfile, "rb") as ifile:
        testcase = pickle.load(ifile)
        case_input, case_output = testcase
    #2. get output.
    output = new_func(**case_input)
    print(f"{output.keys()}")
    print(f"{case_output.keys()}")
    # print(f"{output[0][0].shape}, {output[0][0].dtype}, {output[0][0].device}")
    # print(f"{case_output[0][0].shape}, {case_output[0][0].dtype}, {case_output[0][0].device}")
    #3. return pass/fail.
    return eq_test(output, case_output)


func_path_dict = {
    "get_filter_weights": Path("MNIST-rot/logs/get_filter_weights.iodict"),
}



if __name__ == "__main__":
    val = test_function(func_path_dict["get_filter_weights"], get_filter_weights, compare_filter_weight_output)
    print(f"get_filter_weights: {'Passed!' if val else 'Failed!'}")