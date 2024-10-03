struct __attribute__((__packed__)) serial_control_in {
    float pos_x, pos_y, pos_z;
    float vel_body_x, vel_body_y, vel_body_z;
    float gyro_x, gyro_y, gyro_z;
    float roll, pitch, yaw;
    //CHECKSUM
    uint8_t checksum_in;
};


