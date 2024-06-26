#include "../lif.h"

// Addition/decay/reset constants as const array here, use pointer in
// configuration struct

float const betas_2[] = { 1.0082e+00,  6.2574e-02,  4.9385e-01, -1.4174e-03, -4.5299e-03,-1.5275e-04, -2.0817e-03,  6.5239e-01,  5.4201e-01,  1.0005e+00,2.9582e-01,  1.0345e+00,  1.0013e+00,  1.0197e+00,  4.3745e-01,1.0014e+00,  1.0022e+00, -4.3209e-03, -4.1545e-03,  1.0071e+00,1.0177e+00, -1.2882e-03,  6.4947e-01,  1.0017e+00, -4.3685e-03,7.4937e-01,  8.1458e-01,  4.7681e-01,  4.6583e-01,  4.9897e-01,1.0048e+00,  1.0017e+00};
float const th_2[] = {0.2733, 0.8881, 0.9990, 0.1113, 0.9040, 0.6091, 0.9756, 0.7200, 0.6895,
        0.5492, 0.4169, 0.4248, 0.8331, 0.0981, 0.2509, 0.6279, 0.6495, 0.9585,
        0.3359, 0.7501, 0.7456, 0.6725, 0.8198, 0.1423, 0.8799, 0.3766, 0.3998,
        0.3351, 0.4040, 0.3879, 0.5430, 0.3609};

// type, size, a_v, a_th, a_t, d_v, d_th, d_t, v_rest, th_rest
lif_conf const lif_2 = {32, betas_2, th_2, SLIF};
