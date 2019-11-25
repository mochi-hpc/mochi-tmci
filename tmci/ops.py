import tensorflow as tf
import os
import glob

__tmci_directory = os.path.dirname(__file__) + '/..'
__tmci_library   = glob.glob(__tmci_directory + '/_tmci.*.so')
if len(__tmci_library) == 0:
    raise ImportError('Could not find TMCI shared library')
if len(__tmci_library) > 1:
    raise ImportError('Found multiple candidate libraries for TMCI operations')
__tmci_library = __tmci_library[0]

__tmci_operations = tf.load_op_library(__tmci_library)

_checkpoint = __tmci_operations.tmci_checkpoint
#_restore    = __tmci_operations.tmci_restore
