#include "../drone_lander.h"

// Include child structs
#include "test_connection_conf_hidout.h"
#include "test_connection_conf_inhid.h"
#include "test_connection_conf_hidhid.h"
#include "test_lif_conf_1.h"
#include "test_lif_conf_2.h"
#include "test_lif_conf_3.h"

NetworkConf const lander_conf = {1,7,32,32,
  &conf_inhid, &conf_inhidlif, &conf_hidhid, &conf_hidhidlif, &conf_hidout, &conf_hidoutlif};