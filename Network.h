#pragma once

#include "Connection.h"
#include "Neuron.h"
#include "Encode.h"

// Struct that defines a network of two spiking layers
typedef struct Network {
  // Input, encoded input, hidden and output layer sizes
  int in_size, hid_size, out_size;
  // Type (1: LIF, 2: InputALIF, ...)
  int type;
  // placeholder for output and output decay
  float out, tau_out;
  // Encoding input
  Encode *enc;
  // Connection input -> hidden
  Connection *inhid;
  // Hidden neurons
  Neuron *hid;
  // Connection hidden -> output
  Connection *hidout;
} Network;

// Struct that holds the configuration of a two-layer network
// To be used when loading parameters from a header file
typedef struct NetworkConf {
  // Input, encoded input, hidden and output layer sizes
  int const in_size, hid_size, out_size;
  // Type
  int const type;
  // Encoding
  EncodeConf const *enc;
  // Connection input -> hidden
  ConnectionConf const *inhid;
  // Hidden neurons
  NeuronConf const *hid;
  // Connection hidden -> output
  ConnectionConf const *hidout;
  // Output LI decay
  const float tau_out;
} NetworkConf;

// Build network: calls build functions for children
Network build_network(int const in_size, int const hid_size, int const out_size);

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
float forward_network(Network *net);