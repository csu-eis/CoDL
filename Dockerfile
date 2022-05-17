
FROM ubuntu:16.04

MAINTAINER CoDL

RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends apt-utils unzip wget cmake && \
    apt-get install -y --no-install-recommends default-jre default-jdk && \
    apt-get install -y git && \
    apt-get install -y libtinfo5 && \
    apt-get install -y libfreetype6 libfreetype6-dev && \
    apt-get install -y software-properties-common

RUN \
    add-apt-repository -y ppa:jblgf0/python && \
    apt-get update && \
    apt-get install -y python3.6 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.6-tk

RUN \
    python -m pip install jinja2==2.10.1 && \
    python -m pip install markupsafe==2.0.1 && \
    python -m pip install numpy==1.19.2 && \
    python -m pip install pandas==1.0.5 && \
    python -m pip install scipy==1.5.2 && \
    python -m pip install scikit-learn==0.24.2 && \
    python -m pip install rich==3.3.1 && \
    python -m pip install kiwisolver==1.3.1 && \
    python -m pip install matplotlib==3.1.1 && \
    python -m pip install confuse==1.5.0

RUN \
    wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip && \
    unzip -q platform-tools-latest-linux.zip && \
    rm platform-tools-latest-linux.zip && \
    mv platform-tools /root

RUN \
    wget https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip && \
    unzip -q android-ndk-r17c-linux-x86_64.zip && \
    rm android-ndk-r17c-linux-x86_64.zip && \
    mv android-ndk-r17c /root

RUN \
    wget https://github.com/bazelbuild/bazel/releases/download/0.13.0/bazel-0.13.0-installer-linux-x86_64.sh && \
    chmod +x bazel-0.13.0-installer-linux-x86_64.sh && \
    ./bazel-0.13.0-installer-linux-x86_64.sh && \
    rm bazel-0.13.0-installer-linux-x86_64.sh

ENV ANDROID_NDK_HOME /root/android-ndk-r17c
ENV PATH /root/bin:$PATH
ENV PATH /root/platform-tools:$PATH
ENV HOME /root

WORKDIR /root

CMD ["/bin/bash"]

COPY codl-mobile /root/codl-mobile
COPY codl-eval-tools /root/codl-eval-tools
