struct __attribute__((__packed__)) serial_control_out {
    //Motor commands
    int16_t motor_1, motor_2, motor_3, motor_4;
    //CHECKSUM
    uint8_t checksum_out;
};