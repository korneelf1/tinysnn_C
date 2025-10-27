// Include child structs
#include "test_controller_inhid_file.h"
#include "test_controller_hid_1_file.h"
#include "test_controller_hidhid1_file.h"
#include "test_controller_hid_2_file.h"
#include "test_controller_hidout_file.h"

NetworkControllerConf_Korneel const conf = {18, 256, 128, 4, 3, &conf_inhid, &conf_hid_1, &conf_hidhid1, &conf_hid_2, &conf_hidout};