#!/bin/bash

set -eux

poetry run flake8
poetry run npx pyright .
poetry run bandit . --recursive --ini ../../.bandit
poetry run pytest
