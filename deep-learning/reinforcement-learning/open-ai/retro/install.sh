#!/bin/bash

git clone --recursive https://github.com/openai/retro.git gym-retro
cd gym-retro
pip install -e .

git submodule deinit -f --all
rm -rf .git/modules
git submodule update --init

python ./scripts/import_sega_classics.py # Steam Username: muhamuttaqien # Steam Password (leave blank if cached): angga228840