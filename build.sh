#!/usr/bin/env bash
set -e

export WRF_CHEM=0
export WRF_KPP=0
export EM_CORE=1
export NETCDF4=1
export YACC='/usr/bin/yacc -d'
export FLEX_LIB_DIR='/usl/lib/'
export NETCDF=/opt/wrf
export LD_LIBRARY_PATH=/usr/lib
export JASPERLIB=/opt/wrf/lib
export JASPERINC=/opt/wrf/include
export HDF5=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
export LDFLAGS='-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/usr/lib'
export CPPFLAGS='-I/usr/include/hdf5/openmpi -I/usr/include'

git clone --branch v4.2 --depth 1 https://github.com/wrf-model/WRF.git
cd WRF
sed -i 's#  export USENETCDF=$USENETCDF.*#  export USENETCDF="-lnetcdf"#
        s#  export USENETCDFF=$USENETCDFF.*#  export USENETCDFF="-lnetcdff"#' configure
sed -i '242s/.*/ $I_really_want_to_output_grib2_from_WRF = "TRUE" ;/
        405s/.*/  $response = 34 ;/
        667s/.*/  $response = 1 ;/' arch/Config.pl 
./configure
./compile em_real | tee /tmp/wrf_build.log
grep -q "Problems building executables, look for errors in the build log" /tmp/wrf_build.log && { rm /tmp/wrf_build.log; exit 1; }
rm /tmp/wrf_build.log

cd ..
git clone https://github.com/wrf-model/WPS.git
cd WPS
sed -i '117s/.*/$validreponse = 3 ;/' arch/Config.pl
echo -e "3\n" | ./configure -os Linux && ./compile
ln -sf ./ungrib/Variable_Tables/Vtable.GFS ./Vtable