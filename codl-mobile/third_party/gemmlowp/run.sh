task=$1
ip=$2

export ANDROID_NDK_HOME=/home/tclab/bangwhe/codl/env/android-ndk-r17c/

cd /home/tclab/bangwhe/mace-fcver
bazel build --config android --config optimization third_party/gemmlowp:$task --cpu=arm64-v8a

adb connect $ip

adb -s $ip push bazel-bin/third_party/gemmlowp/$task /data/local/tmp/bangwhe
adb -s $ip shell /data/local/tmp/bangwhe/$task --num_rows 256 --num_depth 4096 --num_cols 1024
adb -s $ip shell /data/local/tmp/bangwhe/$task --num_rows 256 --num_depth 1024 --num_cols 4096
adb -s $ip shell /data/local/tmp/bangwhe/$task --num_rows 1024 --num_depth 1024 --num_cols 4096
adb -s $ip shell /data/local/tmp/bangwhe/$task --num_rows 1024 --num_depth 4096 --num_cols 4096
adb -s $ip shell /data/local/tmp/bangwhe/$task --num_rows 4096 --num_depth 4096 --num_cols 4096
