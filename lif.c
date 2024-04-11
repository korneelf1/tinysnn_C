#include "lif.h"
#include <stdlib.h>

lif build_lif(int const size){
    lif lif;
    // Set type
    lif.type = SLIF; // SLIF is spiking NLIF is not spiking
    lif.size = size;
    // lif.beta = beta;
    // lif.thresholds = thresholds;
    // lif.bias = bias;
    // lif.weights = weights;
    lif.beta = malloc(size * sizeof(*lif.beta));
    lif.thresholds = malloc(size * sizeof(*lif.thresholds));
    // lif.bias = malloc(out_size * sizeof(*lif.bias));
    // lif.weights = malloc(in_size*out_size * sizeof(*lif.weights));

    lif.states = calloc(size, sizeof(*lif.states));
    lif.output = calloc(size, sizeof(*lif.output));
    return lif;
};

void reset_lif(lif* neuron) {
    for (int i=0;i<neuron->size;i++){
        neuron->states[i] = 0.0f;
    }
};

float* update_lif(lif* neuron, float* input) {
    for (int i=0;i<neuron->size;i++){
        neuron->states[i] = neuron->states[i] * neuron->beta[i] + input[i];

        if (neuron->type == NLIF){
            neuron->output[i] = neuron->states[i];
        } else{
        if (neuron->states[i] > neuron->thresholds[i]){
            neuron->states[i] = neuron->states[i] - neuron->thresholds[i];
            neuron->output[i] = 1.0f;
        } else {
            neuron->output[i] = 0.0f;
        }}
    }
    return neuron->output;
};

void destroy_lif(lif* neuron) {
    free(neuron->states);
    free(neuron->size);
    free(neuron->beta);
    free(neuron->thresholds);


    free(neuron);
};

void load_lif_from_conf(lif *neuron, lif_conf const *conf){
    neuron->size = conf->size;
    // neuron->out_size = conf->out_size;
    neuron->beta = conf->beta;
    neuron->thresholds = conf->thresholds;
//     neuron->bias = conf->bias;
//     neuron->weights = conf->weights;
};