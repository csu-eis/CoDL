#!/bin/sh

codegen_models_path="mace/codegen/models"

exitWithMessageFunc() {
  echo "[INFO] Exit"
  exit
}

removeDirectoryFunc() {
  directory_path=$1
  if [ -d $directory_path ]
  then
    echo "Remove $directory_path"
    sudo rm -rf $directory_path
  fi
}

if [ -d ${codegen_models_path} ]
then
  echo "[ERROR] Path mace/codegen/models/ exists, we suggest you remove it to save space."
  #ls -l ./mace/codegen/models/

  read -p "[INFO] Remove mace/codegen/models/? (y/n): " bflag
  if [ "$bflag" != "y" ]
  then
    exitWithMessageFunc
  fi

  #rm -rf ./mace/codegen/models/
  removeDirectoryFunc "./mace/codegen/models/"
fi

read -p "[INFO] Mode (1-Temp/2-GitProject): " bmode

createTemporaryDirectoryFunc() {
  echo "[INFO] Create temporary directory ..."

  mkdir -p temp/

  sudo rm -rf temp/mace-master/
  mkdir -p temp/mace-master/

  BACKUP_DIR=temp/mace-master/
}

removePyCacheFunc() {
  removeDirectoryFunc "./tools/dana/__pycache__"
  removeDirectoryFunc "./tools/__pycache__"
  removeDirectoryFunc "./tools/python/py_proto/__pycache__"
  removeDirectoryFunc "./tools/python/quantize/__pycache__"
  removeDirectoryFunc "./tools/python/__pycache__"
  removeDirectoryFunc "./tools/python/utils/__pycache__"
  removeDirectoryFunc "./tools/python/visualize/__pycache__"
  removeDirectoryFunc "./tools/python/transform/__pycache__"
  removeDirectoryFunc "./mace/python/tools/__pycache__"
}

copyFilesFunc() {
  BACKUP_DIR=$1

  echo "[INFO] Copy files to $BACKUP_DIR (password may be required) ..."

  sudo cp -r cmake/ $BACKUP_DIR
  sudo cp -r docker/ $BACKUP_DIR
  sudo cp -r docs/ $BACKUP_DIR
  sudo cp -r examples/ $BACKUP_DIR
  sudo cp -r include/ $BACKUP_DIR
  sudo cp -r mace/ $BACKUP_DIR
  sudo cp -r repository/ $BACKUP_DIR
  sudo cp -r setup/ $BACKUP_DIR
  sudo cp -r test/ $BACKUP_DIR
  sudo cp -r third_party/ $BACKUP_DIR
  sudo cp -r tools/ $BACKUP_DIR

  sudo cp *.bazel $BACKUP_DIR
  sudo cp *.md $BACKUP_DIR
  sudo cp LICENSE $BACKUP_DIR
  sudo cp WORKSPACE $BACKUP_DIR
  sudo cp .gitignore $BACKUP_DIR
}

createZipFileFunc() {
  echo "[INFO] Create zip file ..."

  cd temp/
  tar -czf mace-master-backup.tar.gz mace-master/
  cd ..
}

if [ "$bmode" == "1" ]
then
  createTemporaryDirectoryFunc

  removePyCacheFunc

  copyFilesFunc $BACKUP_DIR

  createZipFileFunc

  echo "[INFO] Backup MACE successfully. See file ./temp/mace-master-backup.tar.gz."
fi

if [ "$bmode" == "2" ]
then
  BACKUP_DIR=$HOME/MyProjects-201019/mace-fcver/
  read -p "[INFO] Is git project directory $BACKUP_DIR (y/N)?: " bflag
  if [ "$bflag" != "y" ]
  then
    exitWithMessageFunc
  fi

  removePyCacheFunc

  copyFilesFunc $BACKUP_DIR

  echo "[INFO] OK"
fi
