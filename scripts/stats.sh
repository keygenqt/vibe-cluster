#!/bin/bash

source "$(dirname "$0")/common.sh"
setup_root

print_pretty "Project stats (tokei)..." "info"
tokei -e '*.json' -e '*.svg' -e '*.txt'
