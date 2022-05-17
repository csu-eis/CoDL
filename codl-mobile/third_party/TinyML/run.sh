ip=$1

cd /home/tclab/bangwhe/mace-fcver
bazel build --config android --config optimization third_party/TinyML:LinearRegression_main --cpu=arm64-v8a

adb connect $ip

cd bazel-bin/third_party/TinyML
adb -s $ip push LinearRegression_main /data/local/tmp/bangwhe
adb -s $ip shell /data/local/tmp/bangwhe/LinearRegression_main