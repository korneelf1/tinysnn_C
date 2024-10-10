struct __attribute__((__packed__)) serial_control_in {
    float pos_x, pos_y, pos_z;
    float vel_x, vel_y, vel_z;
    float gyro_x, gyro_y, gyro_z;
    float orient_1, orient_2, orient_3, orient_4, orient_5, orient_6, orient_7, orient_8, orient_9;
    //CHECKSUM
    uint8_t checksum_in;
};


