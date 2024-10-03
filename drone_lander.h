#pragma once

#include "Connection.h"
#include "lif.h"

// Struct that defines a network of one spiking layer
typedef struct Network {
  // Input, encoded input and output layer sizes
  int in_size, out_size, hid_1_size, hid_2_size;
// Two input place holders: one for scalar values
  // and one for outputs (size outsize)
  float *in, *out, *out_hid_1, *out_hid_2;
  // Connection input -> output
  Connection *inhid;
  Connection *hidhid;
  Connection *hidout;
  // Output neurons
  lif *inhidlif;
  lif *hidhidlif;
  lif *hidoutlif;
} Network;

typedef struct NetworkConf {
  int const in_size, out_size, hid_1_size, hid_2_size;
    // Connection input -> hidden
  ConnectionConf const *conf_inhid;
  // Hidden neurons
  lif_conf const *conf_inhidlif;
  // Connection hidden -> hidden
  ConnectionConf const *conf_hidhid;
  // Hidden neurons
  lif_conf const *conf_hidhidlif;
  // Connection hidden -> output
  ConnectionConf const *conf_hidout;
  // Output neurons
  lif_conf const *conf_hidoutlif;
} NetworkConf;


// Build network: calls build functions for children
Network build_network(int const in_size, int const hid_1_size, int const hid_2_size, int const out_size);

// Init network: calls init functions for children
void init_network(Network *net);

// Reset network: calls reset functions for children
void reset_network(Network *net);

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(Network *net, NetworkConf const *conf);

// Free allocated memory for network and call free functions for children
void free_network(Network *net);

// Print network parameters (for debugging purposes)
void print_network(Network const *net);

// Forward network and call forward functions for children
// Encoding and decoding inside
int forward_network(Network *net);
