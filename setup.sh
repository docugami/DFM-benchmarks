#!/bin/bash

set -eux

# Install dependencies
sudo apt-get update
sudo apt-get install nodejs npm
curl -sSL https://install.python-poetry.org | python3 -

poetry install
