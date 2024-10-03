#include "drone_lander.h"
#include "Connection.h"
#include "lif.h"
#include "functional.h"
#include "softmax.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build network: calls build functions for children
Network build_network(int const in_size, int const hid_1_size,
                      int const hid_2_size, int const out_size) {
  // Network struct
  Network net;



  // Set sizes
  // Output size has to be 1
//   if (out_size != 1) {
//     printf("Network output size should be 1!\n");
//     exit(1);
//   }
  net.in_size = in_size;
  net.out_size = out_size;
  net.hid_1_size = hid_1_size;
  net.hid_2_size = hid_2_size;

  // Allocate memory for input placeholders, place cell centers and underlying
  // neurons and connections
  net.in = calloc(in_size, sizeof(*net.in));
  net.out = calloc(out_size, sizeof(*net.out));
//   net.outpotentials = calloc(out_size, sizeof(*net.outpotentials));
  net.out_hid_1 = calloc(hid_1_size, sizeof(*net.out_hid_1));
  net.out_hid_2 = calloc(hid_2_size, sizeof(*net.out_hid_2));
  // TODO: is this the best way to do this? Or let network struct consist of
  //  actual structs instead of pointers to structs?
  //  LIFS:
  net.inhidlif = malloc(sizeof(*net.inhidlif));
  net.hidhidlif = malloc(sizeof(*net.hidhidlif));
  net.hidoutlif = malloc(sizeof(*net.hidoutlif));
  //  Connections
  net.inhid = malloc(sizeof(*net.inhid));
  net.hidhid = malloc(sizeof(*net.hidhid));
  net.hidout = malloc(sizeof(*net.hidout));
//   net.out = malloc(sizeof(*net.out));

  // Call build functions for underlying neurons and connections
  *net.inhid = build_connection(hid_1_size, in_size);
  *net.inhidlif = build_lif(hid_1_size);
  *net.hidhid = build_connection(hid_2_size, hid_1_size);
  *net.hidhidlif = build_lif(hid_2_size);
  *net.hidout = build_connection(out_size, hid_2_size);
  *net.hidoutlif = build_lif(out_size);

  return net;
}

// Init network: calls init functions for children
void init_network(Network *net) {
  // Loop over input placeholders
  for (int i = 0; i < net->in_size; i++) {
    net->in[i] = 0.0f;
  }

  for (int i = 0; i < net->out_size; i++) {
    net->out[i] = 0.0f;
  }

  for (int i = 0; i < net->hid_1_size; i++) {
    net->out_hid_1[i] = 0.0f;
  }

  for (int i = 0; i < net->hid_2_size; i++) {
    net->out_hid_2[i] = 0.0f;
  }

  // Call init functions for children
  init_connection(net->inhid);
  reset_lif(net->inhidlif);
  init_connection(net->hidhid);
  reset_lif(net->hidhidlif);
  init_connection(net->hidout);
  reset_lif(net->hidoutlif);
}

// Reset network: calls reset functions for children
void reset_network(Network *net) {
  reset_connection(net->inhid);
  reset_lif(net->inhidlif);
  reset_connection(net->hidhid);
  reset_lif(net->hidhidlif);
  reset_connection(net->hidout);
  reset_lif(net->hidoutlif);
}

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(Network *net, NetworkConf const *conf) {
  // Check shapes
  if ((net->in_size != conf->in_size) ||
      (net->out_size != conf->out_size)) {
    printf(
        "Network has a different shape than specified in the NetworkConf!\n");
    exit(1);
  }

  // Connection input -> hidden
  load_connection_from_header(net->inhid, conf->conf_inhid);
  // Hidden neuron
  load_lif_from_conf(net->inhidlif, conf->conf_inhidlif);
  // Connection hidden -> hidden
  load_connection_from_header(net->hidhid, conf->conf_hidhid);
  // Hidden neuron
  load_lif_from_conf(net->hidhidlif, conf->conf_hidhidlif);
  // Connection hidden -> output
  load_connection_from_header(net->hidout, conf->conf_hidout);
  // Output neuron
  load_lif_from_conf(net->hidoutlif, conf->conf_hidoutlif);
}

// Free allocated memory for network and call free functions for children
void free_network(Network *net) {
  // Call free functions for children
  // Freeing in a bottom-up manner
  // TODO: or should we call this before freeing the network struct members?
  free_connection(net->inhid);
  destroy_lif(net->inhidlif);
  free_connection(net->hidhid);
  destroy_lif(net->hidhidlif);
  free_connection(net->hidout);
  destroy_lif(net->hidoutlif);
  // calloc() was used for input placeholders and underlying neurons and
  // connections
  free(net->in);
//   free(net->in_size);
//   free(net->out_size);
//   free(net->hid_1_size);
//   free(net->hid_2_size);
  free(net->out);
  free(net->out_hid_1);
  free(net->out_hid_2);
}

// Print network parameters (for debugging purposes)
void print_network(Network const *net) {


  // Input layer
  printf("Input layer (raw):\n");
  print_array_1d(net->in_size, net->in);

//   // Connection input -> hidden
//   printf("Connection weights input -> hidden:\n");
//   print_array_2d(net->hid_1_size, net->in_enc_size, net->inhid->w);

//   // Hidden layer
//   print_neuron(net->hid);

//   // Connection hidden -> output
//   printf("Connection weights hidden -> output:\n");
//   print_array_2d(net->out_size, net->hid_size, net->hidout->w);

  // Output layer
//   print_neuron(net->out);
}


// Forward network and call forward functions for children
// Encoding and decoding inside
// TODO: but we still need to check the size of the array we put in net->in
int forward_network(Network *net) {
  // Call forward functions for children
  forward_connection(net->inhid, net->out_hid_1, net->in);
  update_lif(net->inhidlif, net->out_hid_1);
  forward_connection(net->hidhid, net->out_hid_2, net->inhidlif->output);
  update_lif(net->hidhidlif, net->out_hid_2);
  forward_connection(net->hidout, net->out, net->hidhidlif->output);
  update_lif(net->hidoutlif, net->out);
  
  return findMaxIndex(net->hidoutlif->states, net->out_size);
}
