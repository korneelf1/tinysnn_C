#include "functional.h"
#include "drone_lander.h"

// Header file containing parameters
#include "param/test_lander_conf.h"

#include <stdio.h>

// Test network forward functions
// Test network initialization functions
int main() {
  // Build network
  Network net = build_network(lander_conf.in_size, lander_conf.hid_1_size, lander_conf.hid_2_size,
                              lander_conf.out_size);
  // Init network
  init_network(&net);

    // Set input to network
  for (int i = 0; i < lander_conf.in_size; i++) {
    net.in[i] = 2.0f;
  }
  // Load network parameters from header file
  load_network_from_header(&net, &lander_conf);
  reset_network(&net);

  // Forward network
  float output = forward_network(&net);
  float output2 = forward_network(&net);
  float output3 = forward_network(&net);
  float output4 = forward_network(&net);
  float output5 = forward_network(&net);

  // Print network state
  printf("\nFirst run:\n\n");
  // Print output layer trace
  printf("Output layer 1:\n");
  print_array_1d(net.hid_1_size, net.out_hid_1);
  printf("Output LIFS 1:\n");
  print_array_1d(net.hid_1_size, net.hidhidlif->output);
  // Print output layer trace
  printf("Output layer 2:\n");
  print_array_1d(net.hid_2_size, net.out_hid_2);
  printf("Output LIFS 2:\n");
  print_array_1d(net.hid_2_size, net.hidoutlif->output);
  // Print output layer trace
  printf("Output layer 3:\n");
  print_array_1d(net.out_size, net.hidoutlif->output);
  // Print output layer trace
  printf("States layer 3:\n");
  print_array_1d(net.out_size, net.hidoutlif->states);
  // Print output layer trace
  printf("Output layer trace:\n");
  print_array_1d(net.out_size, net.out);

  // Print output layer trace
  printf("Output layer trace:\n");
  print_array_1d(net.out_size, net.out);
  // print the output of the network
  printf("Output Action: %f\n", output);

  // print results for 5th run
  printf("\nFifth run:\n\n");
  // Print output layer trace
  printf("Output layer 1:\n");
  print_array_1d(net.hid_1_size, net.out_hid_1);
  printf("Output LIFS 1:\n");
  print_array_1d(net.hid_1_size, net.hidhidlif->output);
  // Print output layer trace
  printf("Output layer 2:\n");
  print_array_1d(net.hid_2_size, net.out_hid_2);
  printf("Output LIFS 2:\n");
  print_array_1d(net.hid_2_size, net.hidoutlif->output);
  // Print output layer trace
  printf("Output layer 3:\n");
  print_array_1d(net.out_size, net.hidoutlif->output);
  // Print output layer trace
  printf("States layer 3:\n");
  print_array_1d(net.out_size, net.hidoutlif->states);
  // Print output layer trace
  printf("Output layer trace:\n");
  print_array_1d(net.out_size, net.out);

  // Print output layer trace
  printf("Output layer trace:\n");
  print_array_1d(net.out_size, net.out);
// print the output of the network
  printf("Output Action: %f\n", output5);

  // Free network memory again
  free_network(&net);

  return 0;
}