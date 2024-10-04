make clean
cd external/crazyflie-firmware
make cf2_defconfig
cd ../../
make
make cload