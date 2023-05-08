#!/bin/bash

set -eux

# install system dependencies for general development
apt update
apt install nodejs npm
npm install

poetry install
