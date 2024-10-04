struct __attribute__((__packed__)) serial_control_in {
    float pos_x, pos_y, pos_z;
    float vel_body_x, vel_body_y, vel_body_z;
    float gyro_x, gyro_y, gyro_z;
    float quat_w, quat_x, quat_y, quat_z;
    //CHECKSUM
    uint8_t checksum_in;
};


