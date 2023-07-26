#include "../../functional.h"
#include "../../Connection.h"
  
// Weights as const array here, use pointer in configuration struct
float const w_inhid_prop[] = {0.530583f, 0.530583f, 0.525319f, 0.525319f, 0.495098f, 0.495098f, 0.474716f, 0.474716f, 0.526873f, 0.526873f, 0.552026f, 0.552026f, 0.573030f, 0.573030f, 0.544698f, 0.544698f, 0.513898f, 0.513898f, 0.565177f, 0.565177f, 0.506303f, 0.506303f, 0.575095f, 0.575095f, 0.539443f, 0.539443f, 0.549437f, 0.549437f, 0.539470f, 0.539470f, 0.591215f, 0.591215f, 0.567731f, 0.567731f, 0.583635f, 0.583635f, 0.540199f, 0.540199f, 0.533537f, 0.533537f, 0.514221f, 0.514221f, 0.577672f, 0.577672f, 0.486782f, 0.486782f, 0.564851f, 0.564851f, 0.608170f, 0.608170f, 0.463729f, 0.463729f, 0.521568f, 0.521568f, 0.479287f, 0.479287f, 0.493496f, 0.493496f, 0.546968f, 0.546968f, 0.557016f, 0.557016f, 0.592036f, 0.592036f, 0.541470f, 0.541470f, 0.533172f, 0.533172f, 0.503162f, 0.503162f, 0.593221f, 0.593221f, 0.590191f, 0.590191f, 0.511665f, 0.511665f, 0.536369f, 0.536369f, 0.510642f, 0.510642f};

// post, pre, w
ConnectionConf const conf_prop_inhid = {1, 80, w_inhid_prop};
