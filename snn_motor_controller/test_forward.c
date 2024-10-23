#include "NetworkController_Korneel.h"
#include "functional.h"
#include "controller/test_actor_conf.h"

#include <stdio.h>
NetworkController_Korneel net;
float gyro_x = 0.0f;
float gyro_y = 0.0f;
float gyro_z = 0.0f;
float pos_x = 0.0f;
float pos_y = 0.0f;
float pos_z = 0.0f;
float vel_x = 0.0f;
float vel_y = 0.0f;
float vel_z = 0.0f;
float orient_1 = 0.0f;
float orient_2 = 0.0f;
float orient_3 = 0.0f;
float orient_4 = 0.0f;
float orient_5 = 0.0f;
float orient_6 = 0.0f;
float orient_7 = 0.0f;
float orient_8 = 0.0f;
float orient_9 = 0.0f;

float inputs[18];

// Test network forward functions
int main() {
  // Build network
  net = build_network(18, 256,128, 4);
  for (int i = 0; i < 18; i++) {
    inputs[i] = 0.50f;
  }
  init_network(&net);


  // Load network parameters from header file and reset
  load_network_from_header(&net, &conf);
  reset_network(&net);
  // Set input to network
  set_network_input(&net, inputs);

  // Load network parameters from header file
  reset_network(&net);

  for (int j = 0; j < 100; j++) {
    // Forward network
    forward_network(&net);
    // printf("\nOutput after tanh\n");
    // for (int i = 0; i < net.out_size; i++) {
    //   printf("%d: %f\t", i, net.outtanh[i]);
    // }
    // printf("\n");
  }


//   // Print network state
//   printf("\nHeader loading\n\n");
//   // Encoding type
//   printf("Encoding type: %d\n", net.type);
//   printf("Place cell centers:\n");
//   print_array_1d(net.in_enc_size, net.centers);
//   // Decoding scale
//   printf("Decoding scale: %.4f\n\n", net.decoding_scale);
//   // Print network inputs
//   printf("Network inputs:\n");
//   print_array_1d(net.in_size, net.in);
//   // Print encoded network inputs
//   printf("Encoded network inputs:\n");
//   print_array_1d(net.in_enc_size, net.in_enc);
//   // Print input -> hidden weights
//   printf("Input -> hidden weights:\n");
//   print_array_2d(net.hid_size, net.in_enc_size, net.inhid->w);
//   // Print hidden layer inputs
//   printf("Hidden layer inputs:\n");
//   print_array_1d(net.hid_size, net.hid->x);
//   // Print hidden layer voltages
//   printf("Hidden layer voltages:\n");
//   print_array_1d(net.hid_size, net.hid->v);
//   // Print hidden layer thresholds
//   printf("Hidden layer thresholds:\n");
//   print_array_1d(net.hid_size, net.hid->th);
//   // Print hidden layer spikes
//   printf("Hidden layer spikes:\n");
//   print_array_1d_bool(net.hid_size, net.hid->s);
//   // Print hidden layer spike count
//   printf("Hidden layer spike count: %d\n\n", net.hid->s_count);
//   // Print hidden layer trace
//   printf("Hidden layer trace:\n");
//   print_array_1d(net.hid_size, net.hid->t);
//   // Print hidden -> output weights
//   printf("Hidden -> output weights:\n");
//   print_array_2d(net.out_size, net.hid_size, net.hidout->w);
//   // Print output layer inputs
//   printf("Output layer inputs:\n");
//   print_array_1d(net.out_size, net.out->x);
//   // Print output layer voltages
//   printf("Output layer voltages:\n");
//   print_array_1d(net.out_size, net.out->v);
//   // Print output layer thresholds
//   printf("Output layer thresholds:\n");
//   print_array_1d(net.out_size, net.out->th);
//   // Print output layer spikes
//   printf("Output layer spikes:\n");
//   print_array_1d_bool(net.out_size, net.out->s);
//   // Print output layer spike count
//   printf("Output layer spike count: %d\n\n", net.out->s_count);
//   // Print output layer trace
//   printf("Output layer trace:\n");
//   print_array_1d(net.out_size, net.out->t);
//   // Print decoded network output
//   printf("Decoded network output:\n");
//   printf("%.2f\n", output);


  // Free network memory again

  free_network(&net);
// 

  return 0;
}