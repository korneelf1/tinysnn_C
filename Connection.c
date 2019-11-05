#include "Connection.h"
#include "functional.h"

#include <stdio.h>
#include <stdlib.h>

// Build connection
Connection build_connection(int const post, int const pre) {
  // Connection struct
  Connection c;

  // Set shape
  c.post = post;
  c.pre = pre;

  // Allocate memory for weight array
  c.w = calloc(c.post * c.pre, sizeof(*c.w));

  return c;
}

// Init/reset connection
void reset_connection(Connection *c) {
  // Loop over weights
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      c->w[i * c->pre + j] = rand() / (float)RAND_MAX;
    }
  }
}

// Load parameters for connection (weights) from text
void load_connection(Connection *c, char const path[]) {
  // Read file containing weights
  FILE *file;
  if ((file = fopen(path, "r")) == NULL) {
    exit(1);
  }
  // Consists of "post" lines of "pre" weights
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      // TODO: does this work?
      fscanf(file, "%f", &c->w[i * c->pre + j]);
    }
  }
  fclose(file);
}

// Free allocated memory for connection
void free_connection(Connection *c) {
  // calloc() was used for weight array
  // Only one call, so only one free (as opposed to other methods for 2D arrays)
  free(c->w);
}

// Forward
void forward_connection(Connection *c, float x[], bool const s[]) {
  // Loop over weights and multiply with spikes
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      x[i] += c->w[i * c->pre + j] * (float)s[j];
    }
  }
}