#!/bin/sh

export ANDROID_PROJECT_PATH=$HOME/MyProjects_200708/MaceTestApplication

copyFileFunc() {
  src_file=$1
  dst_directory=$2

  if [ ! -f "$src_file" ]
  then
    return
  fi

  if [ ! -d "$dst_directory" ]
  then
    return
  fi
  
  cp $src_file $dst_directory
  echo "[INFO] Copy $src_file"
}

echo "[INFO] Android Project: $ANDROID_PROJECT_PATH"
read -p "[INFO] (y/N): " answer
if [ "$answer" != "y" ]
then
  echo "[INFO] Exit"
  exit
fi

if [ ! -d "$ANDROID_PROJECT_PATH" ]
then
  echo "[ERROR] Android project does not exist"
  echo "[INFO] Exit"
  exit
fi

copyFileFunc ./build/include/mace/public/mace.h \
  $ANDROID_PROJECT_PATH/app/src/main/cpp/include/mace/public

copyFileFunc ./build/lib/arm64-v8a/cpu_gpu/libmace.so \
  $ANDROID_PROJECT_PATH/app/src/main/jniLibs/arm64-v8a

echo "[INFO] OK"
