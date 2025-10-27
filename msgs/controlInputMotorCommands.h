struct __attribute__((__packed__)) serial_control_in {
    // Position
    float pos_x;
    float pos_y;
    float pos_z;
    
    // Attitude (quaternion)
    float qw;
    float qx;
    float qy;
    float qz;
    
    // Velocity
    float vel_x;
    float vel_y;
    float vel_z;
    
    // Gyro
    float gyro_x;
    float gyro_y;
    float gyro_z;
    
    // Control flag
    bool warmUp;
    
    // CHECKSUM
    uint8_t checksum_in;
};


