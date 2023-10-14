#!/bin/bash

set -eux

# install system dependencies for general development
apt-get update
apt-get install nodejs npm
npm install

poetry install
