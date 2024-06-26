#include "../lif.h"

// Addition/decay/reset constants as const array here, use pointer in
// configuration struct

float const betas_1[] = {-5.0110e-04,  3.9099e-01,  9.8174e-02,  4.9266e-01,  1.4357e-01, 4.5383e-01,  1.0059e+00,  6.3891e-01,  4.7427e-01,  7.1373e-01,6.4809e-01, -6.8452e-03,  1.0131e+00,  2.9987e-01,  1.4174e-01,7.3930e-01, -8.6838e-04,  1.0027e+00,  1.0289e+00, -6.0622e-04,5.0713e-01,  1.0009e+00,  1.0015e+00,  1.0024e+00,  1.0080e+00,3.3709e-01,  3.8320e-01,  4.7710e-01, -2.2778e-03,  1.0194e+00,-2.9756e-03,  1.0006e+00};
float const th_1[] = {0.2671, 0.0605, 0.9955, 0.1458, 0.5549, 0.7777, 0.8410, 0.6095, 0.1820,
        0.2454, 0.7686, 0.3475, 0.9736, 0.3929, 0.0378, 0.1485, 0.6558, 0.7679,
        0.0661, 0.9301, 0.0711, 0.8897, 0.6263, 0.9470, 0.0771, 0.5237, 0.1483,
        0.5623, 0.1011, 0.0547, 0.2705, 0.1786};

// type, size, a_v, a_th, a_t, d_v, d_th, d_t, v_rest, th_rest
lif_conf const lif_1 = {32, betas_1, th_1, SLIF};
