#!/bin/bash
# setup environment

set -e

# get the directory above this script
DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

VENV_DIR=$DIR/venvs/

if [ ! -f $VENV_DIR ]; 
then
    mkdir -p venvs
    python3.12 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

pip install --upgrade pip setuptools
pip install -r $DIR/config/requirements.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# setup depth anything v3
cd $DIR/ && pip install -e .
# pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

