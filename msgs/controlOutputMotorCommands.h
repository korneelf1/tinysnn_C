struct __attribute__((__packed__)) serial_control_out {
    // Motor commands
    float motor_1;
    float motor_2;
    float motor_3;
    float motor_4;
    // CHECKSUM
    uint8_t checksum_out;
};