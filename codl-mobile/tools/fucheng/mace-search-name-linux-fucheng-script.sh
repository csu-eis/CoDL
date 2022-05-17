#!/bin/bash

# Example: bash tools/mace-search-name-linux-fucheng-script.sh mace/ "OpenCLRuntime" *.h

target_path=$1
target_name=$2
target_file_type=$3

find $target_path -name $target_file_type | xargs -i grep -Hn "$target_name" {}
