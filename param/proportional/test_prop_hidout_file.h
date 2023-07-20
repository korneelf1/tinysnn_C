#include "../../Connection.h"

// Weights as const array here, use pointer in configuration struct
float const w_hidout_prop[] = {0.008163f, -0.008163f, 0.007721f, -0.007721f, 0.010192f, -0.010192f, 0.007668f, -0.007668f, 0.006782f, -0.006782f, 0.007435f, -0.007435f, 0.008044f, -0.008044f, 0.009872f, -0.009872f, 0.008099f, -0.008099f, 0.007974f, -0.007974f, 0.009017f, -0.009017f, 0.007854f, -0.007854f, 0.007734f, -0.007734f, 0.010955f, -0.010955f, 0.008104f, -0.008104f, 0.008114f, -0.008114f, 0.008977f, -0.008977f, 0.010543f, -0.010543f, 0.007709f, -0.007709f, 0.012175f, -0.012175f, 0.008486f, -0.008486f, 0.007626f, -0.007626f, 0.012589f, -0.012589f, 0.011825f, -0.011825f, 0.009988f, -0.009988f, 0.007148f, -0.007148f, 0.008183f, -0.008183f, 0.010186f, -0.010186f, 0.010791f, -0.010791f, 0.007746f, -0.007746f, 0.007984f, -0.007984f, 0.006458f, -0.006458f, 0.009916f, -0.009916f, 0.007730f, -0.007730f, 0.008851f, -0.008851f, 0.008178f, -0.008178f, 0.009475f, -0.009475f, 0.008341f, -0.008341f, 0.007677f, -0.007677f, 0.007740f, -0.007740f};

// post, pre, w
ConnectionConf const conf_prop_hidout = {80, 1, w_hidout_prop};
