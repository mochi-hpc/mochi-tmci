from ctypes import cdll

def load(filename):
    if filename in load.libraries:
        return
    load.libraries[filename] = cdll.LoadLibrary(filename)

load.libraries = dict()
