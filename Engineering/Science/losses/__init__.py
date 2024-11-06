""""
This package should contain all the losees. 
If the loss is just a single script, then can be added directly here as a file. Add the following info at the top of the file: original paper, bibtext, link to the original code (if obtained) and a brief description of the loss.
If the loss is a collection of scripts, then it should be added as a folder. The folder should contain an __init__.py and then the codes related to the loss.
In case of the latter, the __init__.py should contain the following info: original paper, bibtext, and a brief description of the loss. If the code was obtained from a repo, that link must be added here as well.

The model should be added to the __all__ list in this file.
They should also be added to the LOSSID dict of this file.
Finally, whether or not this loss requires the negavtive of it to be optimised (e.g. SSIM that needs to be maximised and not minmised), indicate that in the IS_NEG_LOSS dict.

If they are default pytorch losses, or from different library, then only add them them in LOSSID and IS_NEG_LOSS.
"""

__all__ = [] # list of all the losses as strings

LOSSID = {
    -2: "LPIPS: Learned Perceptual Image Patch Similarity (https://arxiv.org/abs/1801.03924)",
    -1: "pLoss: Perceptual Loss (in-house implementation))",
    0: "L1"
}

LOSSID_CLS = {
    0: "L1",
    1: "MSE",
    2: "SmoothL1Loss"
}

IS_NEG_LOSS = {
    -2: False,
    -1: False,
    0: False,
    1: False,
    2: False
}