#include "NetworkController_Korneel.h"
#include "Connection.h"
#include "Neuron.h"
#include "functional.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build network: calls build functions for children
NetworkController_Korneel build_network(int const in_size, int const hid1_size, int const hid2_size, int const hid3_size, int const out_size) {
  // Network struct
  NetworkController_Korneel net;

  // Set sizes
  net.in_size = in_size;
  net.hid1_size = hid1_size;
  net.hid2_size = hid2_size;
  net.hid3_size = hid3_size;
  net.out_size = out_size;

  // Initialize type as LIF
  net.type = 1;

  // Allocate memory for input placeholders and underlying
  // neurons and connections
  net.in = calloc(in_size, sizeof(*net.in));
  net.hid_1_in = calloc(hid1_size, sizeof(*net.hid_1_in));
  net.hid_2_in = calloc(hid2_size, sizeof(*net.hid_2_in));
  net.hid_3_in = calloc(hid3_size, sizeof(*net.hid_3_in));

  // neurons
  net.hid_1 = malloc(sizeof(*net.hid_1));
  net.hid_2 = malloc(sizeof(*net.hid_2));
  net.hid_3 = malloc(sizeof(*net.hid_3));

  // connections
  net.inhid = malloc(sizeof(*net.inhid));
  net.hidhid_1 = malloc(sizeof(*net.hidhid_1));
  net.hidhid_2 = malloc(sizeof(*net.hidhid_2));
  net.hid3out = malloc(sizeof(*net.hid3out));

  net.out = calloc(out_size, sizeof(*net.out));

  // Call build functions for underlying neurons and connections
  *net.inhid = build_connection(in_size, hid1_size);
  *net.hidhid_1 = build_connection(hid1_size, hid2_size);
  *net.hidhid_2 = build_connection(hid2_size, hid3_size);
  *net.hid3out = build_connection(hid3_size, out_size);
  *net.hid_1 = build_neuron(hid1_size);
  *net.hid_2 = build_neuron(hid2_size);
  *net.hid_3 = build_neuron(hid3_size);



  return net;
}

// Init network: calls init functions for children
void init_network(NetworkController_Korneel *net) {
    // Call init functions for children
    init_connection(net->inhid);
    init_connection(net->hidhid_1);
    init_connection(net->hidhid_2);
    init_connection(net->hid3out);
    init_neuron(net->hid_1);
    init_neuron(net->hid_2);
    init_neuron(net->hid_3);

}

// Reset network: calls reset functions for children
void reset_network(NetworkController_Korneel *net) {
  for (int i = 0; i < net->out_size; i++) {
    net->out[i] = 0.0f;
  }
    reset_connection(net->inhid);
    reset_connection(net->hidhid_1);
    reset_connection(net->hidhid_2);
    reset_connection(net->hid3out);
    reset_neuron(net->hid_1);
    reset_neuron(net->hid_2);
    reset_neuron(net->hid_3);

}

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(NetworkController_Korneel *net, NetworkControllerConf_Korneel const *conf) {
  // Check shapes
  if ((net->in_size != conf->in_size) ||
      (net->hid1_size != conf->hid1_size) ||
      (net->hid2_size != conf->hid2_size) ||
      (net->hid3_size != conf->hid3_size) ||
      (net->out_size != conf->out_size)) {
    printf(
        "Network has a different shape than specified in the NetworkConf!\n");
    exit(1);
  }
  // Set type
  net->type = conf->type;

  // Connection input -> hidden
  load_connection_from_header(net->inhid, conf->inhid);
  // Hidden neurons
  load_neuron_from_header(net->hid_1, conf->hid_1);
  // Connection hidden -> hidden
  load_connection_from_header(net->hidhid_1, conf->hidhid_1);
  // Hidden neurons
  load_neuron_from_header(net->hid_2, conf->hid_2);
  // Connection hidden -> hidden
  load_connection_from_header(net->hidhid_2, conf->hidhid_2);
  // Hidden neurons
  load_neuron_from_header(net->hid_3, conf->hid_3);
  // Connection hidden -> output
  load_connection_from_header(net->hid3out, conf->hid3out);
}

// Set the inputs of the controller network with given floats
void set_network_input(NetworkController_Korneel *net, float inputs[]) {
    net->in[0] = inputs[0];
    net->in[1] = inputs[1];
    net->in[2] = inputs[2];
    net->in[3] = inputs[3];
    net->in[4] = inputs[4];
    net->in[5] = inputs[5];
    net->in[6] = inputs[6];
    net->in[7] = inputs[7];
    net->in[8] = inputs[8];
    net->in[9] = inputs[9];
    net->in[10] = inputs[10];
    net->in[11] = inputs[11];
    net->in[12] = inputs[12];
}


// Forward network and call forward functions for children
// Encoding and decoding inside
float* forward_network(NetworkController_Korneel *net) {
//   forward_connection_real(net->inenc, net->enc->x, net->in);
  forward_connection_fast(net->inhid, net->hid_1->x, net->in);
  forward_neuron(net->hid_1);
//   forward_connection(net->enchid, net->hid->x, net->enc->s);
  forward_connection_fast(net->hidhid_1, net->hid_2->x, net->hid_1->s);
  forward_neuron(net->hid_2);
  forward_connection_fast(net->hidhid_2, net->hid_3->x, net->hid_2->s);
  forward_neuron(net->hid_3);
  forward_connection_fast(net->hid3out, net->out, net->hid_3->s);
  
  for (int i = 0; i < net->out_size; i++) {
    net->outtanh[i] = tanh(net->out[i]);
  }
  return net->outtanh;
}


// Print network parameters (for debugging purposes)
void print_network(NetworkController_Korneel const *net) {
  // Input layer
//   printf("Input layer (raw):\n");
//   print_array_1d(net->in_size, net->in);
  printf("Input layer (encoded):\n");

  // Connection input -> hidden
//   printf("Connection weights input -> encoding:\n");
//   print_array_2d(net->enc_size, net->in_size, net->inenc->w);

  // Hidden layer
  print_neuron(net->hid_1);

  // Connection hidden -> output
  printf("Connection weights hidden -> output:\n");
  print_array_2d(net->out_size, net->hid3_size, (const float **)net->hid3out->w);


}


// Free allocated memory for network and call free functions for children
void free_network(NetworkController_Korneel *net) {
  // Call free functions for children
  // Freeing in a bottom-up manner
  // TODO: or should we call this before freeing the network struct members?
free_connection(net->inhid);
free_connection(net->hidhid_1);
free_connection(net->hidhid_2);
free_connection(net->hid3out);
free_neuron(net->hid_1);
free_neuron(net->hid_2);
free_neuron(net->hid_3);

  // calloc() was used for input placeholders and underlying neurons and
  // connections
free(net->inhid);
free(net->hidhid_1);
free(net->hidhid_2);
free(net->hid3out);
free(net->hid_1);
free(net->hid_2);
free(net->hid_3);
free(net->in);
free(net->out);
free(net->outtanh);

    
}