import json
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())


def listify(x): 
    if x is None: 
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def get_files(path, extension):
    """
    Returns all the files in the given path which matches the extension
    """
    extension = listify(extension)
    return [p for p in path.ls() if p.suffix in extension and
            '(' not in p.stem]


def get_images(path, ext='.jpg'):
    """
    Returns a list of paths of all images in the given path.
    """
    return get_files(path, ext)


def get_label_files(path, ext='.txt'):
    """
    Returns a list of paths of all label files.
    """
    return get_files(path, ext)


def get_label(img_path):
    """
    Gets a label from a given image path
    """
    img_name = img_path.stem
    label_name = img_name+'.txt'
    label_path = img_path.parent/label_name
    with open(label_path) as f:
        label = json.load(f)
    return label