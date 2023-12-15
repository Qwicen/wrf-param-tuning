FROM ubuntu:16.04
USER root
RUN apt update
RUN apt install -y software-properties-common
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential gfortran \
    sudo libgrib-api-dev vim libhdf5-openmpi-dev libnetcdf-dev mpich libhdf5-mpich-dev \
    libopenmpi-dev openmpi-bin libnetcdff-dev libnetcdf-dev libjasper-dev \
    libgrib2c-dev libpng-dev netcdf-bin flex-old byacc csh perl curl m4 git \
    python3 python3-numpy python3-pandas python3-boto3
RUN apt install -y wget
RUN apt autoclean && apt autoremove -y && rm -rf /var/lib/{apt,dpkg,cache,log}/
RUN groupadd -r wrfuser && useradd -l -m -r -g wrfuser wrfuser && \
    adduser wrfuser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && ln -sf /usr/lib/openmpi/include/mpif.h /usr/include/
RUN mkdir -p /opt/wrf && ln -s /usr/include /opt/wrf/include \
    && ln -s /usr/lib/x86_64-linux-gnu /opt/wrf/lib
COPY build.sh /home/wrfuser/build.sh
COPY environment.yml /home/wrfuser/
COPY . /home/wrfuser/pipeline
RUN chown wrfuser:wrfuser /home/wrfuser/build.sh
USER wrfuser
RUN cd /home/wrfuser/ && bash build.sh
WORKDIR /home/wrfuser
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN /home/wrfuser/miniconda3/bin/conda env create -f environment.yml
