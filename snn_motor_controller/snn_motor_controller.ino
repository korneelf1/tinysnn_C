#include <Arduino.h>

#include "msgs/controlInputMotorCommands.h"
#include "msgs/controlOutputMotorCommands.h"

extern "C" {
#include "NetworkController_Korneel.h"
#include "functional.h"
#include "controller/test_actor_conf.h"
}

// -------------------------- COMMUNICATION DEFINED VARIABLES-----------------------------
#define COMMUNICATION_SERIAL Serial2
#define COMMUNICATION_SERIAL_BAUD 460800
// #define COMMUNICATION_SERIAL_BAUD 230400

byte START_BYTE_SERIAL_CF = 0x9A;
elapsedMicros last_time_write_to_cf = 0;
int ack_comm_TX_cf = 0;

struct serial_control_in myserial_control_in;
uint8_t serial_cf_msg_buf_in[2 * sizeof(struct serial_control_in)] = { 0 };
uint16_t serial_cf_buf_in_cnt = 0;
int serial_cf_received_packets = 0;
int serial_cf_missed_packets_in = 0;

volatile struct serial_control_out myserial_control_out;
volatile float extra_data_out[255] __attribute__((aligned));

bool sending;
bool receiving = true;

// -------------------------- DEBUG DEFINED VARIABLES-----------------------------
#define DEBUG_serial Serial
#define DEBUG_serial_baud 115200

// -------------------------- CONTROL DEFINED VARIABLES-------------------------------
elapsedMicros timer_count_main = 0;
elapsedMicros timer_network = 0;
elapsedMicros timer_receive = 0;
elapsedMicros timer_send = 0;
int timer_network_outer = 0;
int timer_receive_outer = 0;
int timer_send_outer = 0;
int n_forward_passes = 0;
NetworkController_Korneel controller;

// -------------------------- INPUT DEFINED VARIABLES-----------------------------
float gyro_x = 0.0f;
float gyro_y = 0.0f;
float gyro_z = 0.0f;
float pos_x = 0.0f;
float pos_y = 0.0f;
float pos_z = 0.0f;
float vel_body_x = 0.0f;
float vel_body_y = 0.0f;
float vel_body_z = 0.0f;
float roll = 0.0f;
float pitch = 0.0f;
float yaw = 0.0f;
float inputs[12] = { pos_x, pos_y, pos_z, vel_body_x, vel_body_y, vel_body_z, roll, pitch, yaw, gyro_x, gyro_y, gyro_z };

///////////////////////////////////////////////USER DEFINED FCN///////////////////
static inline int16_t saturateSignedInt16(float in) {
  // don't use INT16_MIN, because later we may negate it, which won't work for that value.
  if (in > INT16_MAX)
    return INT16_MAX;
  else if (in < -INT16_MAX)
    return -INT16_MAX;
  else
    return (int16_t)in;
}

void serialParseMessageIn(void) {
  //Copy received buffer to structure
  memmove(&myserial_control_in, &serial_cf_msg_buf_in[1], sizeof(struct serial_control_in) - 1);
  // DEBUG_serial.write("Correct message received and storing\n");
  //   DEBUG_serial.write("Stored pitch is %i\n", myserial_control_in.pitch);
}

void setInputMessage(void) {
  inputs[0] = myserial_control_in.pos_x;
  inputs[1] = myserial_control_in.pos_y;
  inputs[2] = myserial_control_in.pos_z;
  inputs[3] = myserial_control_in.vel_body_x;
  inputs[4] = myserial_control_in.vel_body_y;
  inputs[5] = myserial_control_in.vel_body_z;
  inputs[6] = myserial_control_in.roll;
  inputs[7] = myserial_control_in.pitch;
  inputs[8] = myserial_control_in.yaw;
  inputs[9] = myserial_control_in.gyro_x;
  inputs[10] = myserial_control_in.gyro_y;
  inputs[11] = myserial_control_in.gyro_z;

  // inputs = [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, roll_target, pitch_target];
  // DEBUG_serial.printf("%f, %f, %f, %f, %f, %f, %f, %f\n", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]);
  set_network_input(&controller, inputs);
}

void setOutputMessage(void) {
  myserial_control_out.motor_1 = saturateSignedInt16(controller.out[0]);
  myserial_control_out.motor_2 = saturateSignedInt16(controller.out[1]);
  myserial_control_out.motor_3 = saturateSignedInt16(controller.out[2]);
  myserial_control_out.motor_4 = saturateSignedInt16(controller.out[3]);
}

void sendCrazyflie(void) {
  //SENDING PACKET

  //Calculate checksum for outbound packet:
  uint8_t *buf_send = (uint8_t *)&myserial_control_out;
  myserial_control_out.checksum_out = 0;
  for (uint16_t i = 0; i < sizeof(struct serial_control_out) - 1; i++) {
    myserial_control_out.checksum_out += buf_send[i];
  }

  //Send out packet to buffer:
  noInterrupts();
  COMMUNICATION_SERIAL.write(START_BYTE_SERIAL_CF);
  COMMUNICATION_SERIAL.write(buf_send, sizeof(struct serial_control_out));
  interrupts();

  last_time_write_to_cf = 0;

  sending = false;
  receiving = true;
}


void receiveCrazyflie(void) {
  //RECEIVING PACKET
  //Collect packets on the buffer if available:
  while (COMMUNICATION_SERIAL.available()) {
    // DEBUG_serial.write("trying to read...\n");

    timer_receive = 0;
    uint8_t serial_cf_byte_in;
    serial_cf_byte_in = COMMUNICATION_SERIAL.read();
    if ((serial_cf_byte_in == START_BYTE_SERIAL_CF) || (serial_cf_buf_in_cnt > 0)) {
      serial_cf_msg_buf_in[serial_cf_buf_in_cnt] = serial_cf_byte_in;
      serial_cf_buf_in_cnt++;
    }
    if (serial_cf_buf_in_cnt > sizeof(struct serial_control_in)) {
      serial_cf_buf_in_cnt = 0;
      uint8_t checksum_in_local = 0;
      for (uint16_t i = 1; i < sizeof(struct serial_control_in); i++) {
        checksum_in_local += serial_cf_msg_buf_in[i];
      }
      if (checksum_in_local == serial_cf_msg_buf_in[sizeof(struct serial_control_in)]) {
        serialParseMessageIn();
        serial_cf_received_packets++;

      } else {
        serial_cf_missed_packets_in++;
        DEBUG_serial.write("Incorrect message\n");
      }
      receiving = false;
      sending = true;
    }
    timer_receive_outer = timer_receive_outer + timer_receive;
    timer_receive = 0;
  }
}

void setup(void) {
  //////////////////SETUP DEBUGGING USB
  DEBUG_serial.begin(DEBUG_serial_baud);


  //////////////////Initialize controller network
  DEBUG_serial.write("Build network\n");
  controller = build_network(13, 256, 256,128, 4);
  DEBUG_serial.write("Init network\n");
  init_network(&controller);


  // Load network parameters from header file and reset
  DEBUG_serial.write("Loading network\n");
  DEBUG_serial.write("\n");
  load_network_from_header(&controller, &conf);
  DEBUG_serial.write("Resetting\n");
  reset_network(&controller);

  //////////////////SETUP CONNECTION WITH CRAZYFLIE
  COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD);
  DEBUG_serial.write("Finished setup\n");
}

///////////////////////////////////////////////////////////LOOP///////////////////
void loop(void) {
  if (receiving) {
    // DEBUG_serial.write("receiving...");
    receiveCrazyflie();
  } else if (sending) {
    // Timer for debugging
    if (timer_count_main > 1000000) {
      DEBUG_serial.printf("Received %i packets over last second\n", serial_cf_received_packets);
      DEBUG_serial.printf("Processing network took %i ms for %i forward passes\n", timer_network_outer / 1000, n_forward_passes);
      DEBUG_serial.printf("Amounts to %i per inference\n", timer_network_outer / n_forward_passes);
      DEBUG_serial.printf("Receiving took %i ms for %i forward passes\n", timer_receive_outer / 1000, n_forward_passes);
      DEBUG_serial.printf("Sending took %i ms for %i forward passes\n", timer_send_outer / 1000, n_forward_passes);
      DEBUG_serial.printf("Last control output x:%d, y:%d, z:%d\n", myserial_control_out.motor_1, myserial_control_out.motor_2, myserial_control_out.motor_3);
      // DEBUG_serial.printf("CPU temp is %f\n", tempmonGetTemp());
      serial_cf_received_packets = 0;
      timer_count_main = 0;
      timer_network_outer = 0;
      timer_receive_outer = 0;
      timer_send_outer = 0;
      n_forward_passes = 0;
    }
    // Set input to network from CF
    // DEBUG_serial.write("Setting input message\n");
    setInputMessage();

    // Reset network if thrust command is zero
    // TODO: Find better solution, otherwise network might be reset mid flight
    // if (myserial_control_in.thrust == 0.0f) {
    //  reset_network(&controller);
//    }

    // Forward network
    timer_network = 0;
    forward_network(&controller);
    timer_network_outer = timer_network_outer + timer_network;
    n_forward_passes++;
    timer_network = 0;

    // Send message via UART to CF
    timer_send = 0;
    sendCrazyflie();
    timer_send_outer = timer_send_outer + timer_send;
    timer_send = 0;




    // roll_integ += controller.out[0] - 5 * controller.out[2];
    // pitch_integ += controller.out[1] + 5 * controller.out[3];

    // Store output message to be sent back to CF
    setOutputMessage();
  }
}