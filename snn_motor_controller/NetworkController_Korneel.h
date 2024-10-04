#pragma once

#include "Connection.h"
#include "Neuron.h"

// Struct that defines a network of two spiking layers
typedef struct NetworkController_Korneel {
  // Input, hidden and output layer sizes
  int in_size, hid1_size, hid2_size, hid3_size, out_size;
  // Type (1: LIF, 2: InputALIF, ...)
  int type;
  // placeholder for input, hidden and output
  float *in, *hid_1_in, *hid_2_in, *hid_3_in, *logits_snn,*out;
  float *outtanh;
  // placeholder for output
//   float *out;
  

  // Connection encoding -> hidden
  Connection *inhid;
  // Hidden neurons
  Neuron *hid_1;
  // Recurrent connection hidden 1 -> hidden 2
  Connection *hidhid_1;
  // Hidden neurons
  Neuron *hid_2;
  // Connection hidden 2 -> hidden 3
  Connection *hidhid_2;
  // Hidden neurons
  Neuron *hid_3;
  // Connection hidden 3 -> out
  Connection *hid3out;


} NetworkController_Korneel;

// Struct that holds the configuration of a two-layer network
// To be used when loading parameters from a header file
typedef struct NetworkControllerConf_Korneel {
    // Input, hidden and output layer sizes
    int const in_size, hid1_size, hid2_size, hid3_size, out_size;
    // Type
    int const type;
    // Connection encoding -> hidden
    ConnectionConf const *inhid;
    // Hidden neurons
    NeuronConf const *hid_1;
    // Recurrent connection hidden 1 -> hidden 2
    ConnectionConf const *hidhid_1;
    // Hidden neurons
    NeuronConf const *hid_2;
    // Connection hidden 2 -> hidden 3
    ConnectionConf const *hidhid_2;
    // Hidden neurons
    NeuronConf const *hid_3;
    // Connection hidden 3 -> out
    ConnectionConf const *hid3out;
} NetworkControllerConf_Korneel;

// Build network: calls build functions for children
NetworkController_Korneel build_network(int const in_size, int const hid1_size, int const hid2_size, int const hid3_size, int const out_size);

// Init network: calls init functions for children
void init_network(NetworkController_Korneel *net);

// Reset network: calls reset functions for children
void reset_network(NetworkController_Korneel *net);

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(NetworkController_Korneel *net, NetworkControllerConf_Korneel const *conf);

// Free allocated memory for network and call free functions for children
void free_network(NetworkController_Korneel *net);

// Print network parameters (for debugging purposes)
void print_network(NetworkController_Korneel const *net);

// Set the inputs of the encoding layer
void set_network_input(NetworkController_Korneel *net, float inputs[]);

// Forward network and call forward functions for children
// Encoding and decoding inside
float* forward_network(NetworkController_Korneel *net);