#!/bin/bash

DST_PATH=$HOME/FCProjects-210125/codl-mobile

function create_dir() {
  mkdir -p $1
}

function remove_dir() {
  dirpath=$1
  if [ -d $dirpath ]; then
    rm -rf $dirpath
  fi
}

function copy_dir() {
  cp -r $1 $2
}

function copy_files_with_type() {
  cp $1/*.$3 $2
}

function copy_file() {
  cp $1 $2
}

echo "Removing target directories..."

remove_dir $DST_PATH/test/codlconv2drun

echo "Copying directories..."

copy_dir cmake $DST_PATH
copy_dir docker $DST_PATH
copy_dir docs $DST_PATH
copy_dir examples $DST_PATH
copy_dir include $DST_PATH
copy_dir mace $DST_PATH
copy_dir repository $DST_PATH
copy_dir setup $DST_PATH
copy_dir test $DST_PATH
copy_dir third_party $DST_PATH
copy_dir tools $DST_PATH

echo "Copying files..."

copy_files_with_type . $DST_PATH bazel
copy_files_with_type . $DST_PATH md
copy_file WORKSPACE $DST_PATH

echo "Removing directories..."

remove_dir $DST_PATH/mace/codegen/models
remove_dir $DST_PATH/test/fucheng
remove_dir $DST_PATH/tools/fucheng

echo "Copy all files OK"
