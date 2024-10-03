#include "../lif.h"

// Addition/decay/reset constants as const array here, use pointer in
// configuration struct

float const betas_3[] = { 0.8798,0.8798,0.8798,0.8798,0.8798,0.8798,0.8798};
float const th_3[] = {10.,10.,10.,10.,10.,10.,10.};

// type, size, a_v, a_th, a_t, d_v, d_th, d_t, v_rest, th_rest
lif_conf const conf_hidoutlif = {7, betas_3, th_3, NLIF};
