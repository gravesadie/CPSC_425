import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from functools import partial
conv2d = partial(convolve2d, mode="same", boundary="symm")

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.

    Args:
        im1: first image
        im2: second image

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape

    # START your code here
    # HINT: You can use the conv2d defined in Line 5 for convolution operations
    # NOTE: You should remove the next three lines while coding
    #Ix = np.empty_like(im1)
    #Iy = np.empty_like(im1)
    #It = np.empty_like(im1)

    # instantiate gradient kernels
    dx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    dy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    # compute gradients in x and y
    Ix = conv2d(im1, dx)
    Iy = conv2d(im1, dy)
    # compute time gradient
    It = im2 - im1
    # END your code here

    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It


def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2.2):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: indicates whether to use Gaussian weighting
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction

    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    # START your code here
    # HINT: You can use the conv2d defined in Line 5 for convolution operations
    # NOTE: You can use either linear algebra knowledge or numpy.linalg.inv() for the matrix inverse
    # NOTE: You should remove the next two lines while coding
    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    # create padded versions of inputs Ix Iy It
    Ix_pad = np.pad(array=Ix, pad_width=patch_size//2, mode='symmetric')
    Iy_pad = np.pad(array=Iy, pad_width=patch_size//2, mode='symmetric')
    It_pad = np.pad(array=It, pad_width=patch_size//2, mode='symmetric')

    # begin looping over pixel positions
    for i in range(Ix.shape[0]):
        for j in range(Iy.shape[1]):
            # get appropriate patches and flatten them
            Ix_patch = Ix_pad[i:i+patch_size, j:j+patch_size].flatten()
            Iy_patch = Iy_pad[i:i+patch_size, j:j+patch_size].flatten()
            It_patch = It_pad[i:i+patch_size, j:j+patch_size].flatten()
            # compute A and b
            A = np.array([Ix_patch, Iy_patch]).T
            b = -np.array(It_patch)
            # compute AtA
            Ix_sq = np.sum(list(map(lambda x: x**2, list(Ix_patch))))
            Iy_sq = np.sum(list(map(lambda y: y**2, list(Iy_patch))))
            Ixy = np.sum(list(map(lambda x, y: x*y, list(Ix_patch), list(Iy_patch))))
            AtA = np.array([[Ix_sq, Ixy],[Ixy, Iy_sq]])
            AtA_inv = np.linalg.inv(AtA)
            # compute result from which u and v will be extracted
            res1 = np.matmul(AtA_inv, A.T)
            res2 = np.matmul(res1, b)
            u_i, v_i = res2[0], res2[1]
            u[i,j] = u_i
            v[i,j] = v_i
    # END your code here

    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v


def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image
        u, v: optical flow in x and y direction

    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape

    # START your code here
    # HINT: You can use the np.meshgrid() function
    # HINT: You can use the interpolate.griddata() function with method='linear' and fill_value=0
    # NOTE: You should remove the next line while coding
    im_warp = np.empty_like(im)

    x, y = im.shape[0], im.shape[1]
    grid = np.meshgrid(x, y)
    # basically idk how to use the u and v values here
    # do I need to create new points in between pixels that correspond to the origin coordinates with dx/dy values added?
    interpolate.griddata(points=grid, values=im, xi=grid, method='linear', fill_value=0)
    # np.meshgrid takes in two ranges/vectors that specify the two ranges of a 2D coordinate grid
    # ex. [0,0.5,1] & [0,1] makes a 3x2 array

    # END your code here

    assert im_warp.shape == im.shape
    return im_warp


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    # START your code here
    # NOTE: You should remove the next line while coding
    d = 0.0
    # END your code here

    assert isinstance(d, float)
    return d
