#!/bin/bash

set -eux

# install system dependencies for general development
sudo apt-get update
sudo apt-get install nodejs npm

poetry install
