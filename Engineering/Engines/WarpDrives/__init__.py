""""
This package should contain all the models. 
If the model is just a single script, then can be added directly here as a file. Add the following info at the top of the file: original paper, bibtext, link to the original code (if obtained) and a brief description of the model.
If the model is a collection of scripts, then it should be added as a folder. The folder should contain an __init__.py and then the codes related to the model.
In case of the latter, the __init__.py should contain the following info: original paper, bibtext, and a brief description of the model. If the code was obtained from a repo, that link must be added here as well.

The model should be added to the __all__ list in this file.
And should be added to the MODELID dict of this file.
"""

__all__ = [] # list of all the models as strings

MODELID = {
    -1: "Debug with Identity",
    0: "Models from pythae",
}