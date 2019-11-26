from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from distutils.command.build_clib import build_clib
from distutils.command.build_ext import build_ext
import json
import os
import os.path
import sys

# Find tensorflow headers and libraries
tf_info = {
        'libraries'      : None,
        'library_dirs'   : None,
        'include_dirs'   : None,
        'extra_cxxflags' : None
        }
try:
    with open('tensorflow.json') as f:
        tf_info = json.loads(f.read())
except:
    pass

if tf_info['libraries'] is None:
    tf_info['libraries'] = ':libtensorflow_framework.so.2'
if tf_info['library_dirs'] is None:
    import tensorflow as tf
    path = os.path.dirname(tf.__file__)
    tf_info['library_dirs'] = [ path + '/../tensorflow_core' ]
if tf_info['include_dirs'] is None:
    import tensorflow as tf
    path = os.path.dirname(tf.__file__)
    tf_info['include_dirs'] = [ path + '/../tensorflow_core/include' ]

cxxflags = ['-std=c++14', '-g']
if tf_info['extra_cxxflags'] is not None:
    cxxflags.extend(tf_info['extra_cxxflags'])

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')

tmci_op_module_libraries = tf_info['libraries']
tmci_op_module_library_dirs = tf_info['library_dirs']
tmci_op_module_include_dirs = tf_info['include_dirs'] + ['.']
tmci_op_module = Extension('_tmci_ops',
        ['tmci/src/checkpoint.cpp',
         'tmci/src/restore.cpp',
         'tmci/src/backend.cpp' ],
        libraries=tmci_op_module_libraries,
        library_dirs=tmci_op_module_library_dirs,
        include_dirs=tmci_op_module_include_dirs,
        extra_compile_args=cxxflags,
        depends=[])

setup(name='tmci',
      version='0.1',
      author='Matthieu Dorier',
      description='''Python library to access TensorFlow tensors memory in C++ for checkpoint/restart''',
      ext_modules=[ tmci_op_module ],
      packages=['tmci'],
      headers=['tmci/src/backend.hpp']
    )
