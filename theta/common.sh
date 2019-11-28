#!/bin/bash

# exit on error
set -e

module swap PrgEnv-intel PrgEnv-gnu
module load cce
module load datascience/tensorflow-2.0
