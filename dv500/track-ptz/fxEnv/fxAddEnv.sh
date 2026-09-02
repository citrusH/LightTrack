#!/bin/bash
set -e

# BASE_URL="http://192.168.101.95:8080/track-bbox/dv500/fxEnv/"
# SCRIPT_SELF=$(basename "$0")

mkdir -p ../fxBuild

# find . -maxdepth 1 ! -name "${SCRIPT_SELF}" ! -path . -exec rm -rf {} +
# rm -rf fxEnv
# mkdir -p fxEnv
# cd fxEnv

# wget \
#     --recursive \
#     --no-parent \
#     --cut-dirs=3 \
#     --no-host-directories \
#     --directory-prefix=. \
#     --reject=html \
#     --continue \
#     "${BASE_URL}"

# find . -name "index.html*" -delete
echo "[fx] dependency environment configuration completed!"