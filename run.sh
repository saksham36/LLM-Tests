#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Set the name for the Docker container
CONTAINER_NAME=test

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SERVER_HOSTNAME="$(uname -n)"

INSTALL_ROOT="${INSTALL_ROOT:-/app/lamini}"

DOCKERFILE_NAME="docker-compose.yaml"

SERVER_HOSTNAME=${SERVER_HOSTNAME} docker compose -f $LOCAL_DIRECTORY/$DOCKERFILE_NAME up -d
