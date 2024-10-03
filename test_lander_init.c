#include "functional.h"
#include "drone_lander.h"

// Header file containing parameters
#include "param/test_lander_conf.h"

#include <stdio.h>

// Test network initialization functions
// Test network initialization functions
int main() {
  // Build network
  Network net = build_network(lander_conf.in_size, lander_conf.hid_1_size, lander_conf.hid_2_size,
                              lander_conf.out_size);
  // Init network
  init_network(&net);

  // Print network parameters before loading
  print_network(&net);

  // Load network parameters from header file
  load_network_from_header(&net, &lander_conf);
  reset_network(&net);

  // Print network parameters after header loading
  printf("\nHeader loading\n\n");
  print_network(&net);

  // Free network memory again
  free_network(&net);

  return 0;
}