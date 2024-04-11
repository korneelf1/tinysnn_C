#pragma once
// Enumeration for neuron types: regular LIF and adaptive LIF
typedef enum NeuronType { NLIF, SLIF } NeuronType;

typedef struct lif {
    NeuronType type;
    int size;
    // int out_size;
    float *beta;
    float *states;
    float *thresholds;

    // it is a fused operator with next layer for speedup
    // float *bias;
    float *output;
    // float *weights;
} lif;

typedef struct lif_conf{
    int const size;
    float const *beta;
    float const *thresholds;
    NeuronType const type;
    // float const *bias;
    // float const *weights;
} lif_conf;

lif build_lif(int const size);

void destroy_lif(lif* neuron);

void reset_lif(lif* neuron);

float* update_lif(lif* neuron, float* input);

void load_lif_from_conf(lif *neuron, lif_conf const *conf);
