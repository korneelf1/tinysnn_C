struct __attribute__((__packed__)) serial_control_in {
    float pos_x, pos_y, pos_z;
    float qw, qx, qy, qz;
    float vel_x, vel_y, vel_z;
    float gyro_x, gyro_y, gyro_z;

    bool warmUp;
    //CHECKSUM
    uint8_t checksum_in;
};


