#!/bin/bash

set -eux

poetry install

poetry run flake8
poetry run npx pyright .
poetry run bandit . --recursive --ini ./.bandit
