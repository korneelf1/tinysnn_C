// Include child structs
#include "test_controller_inhid_file.h"
#include "test_controller_hid_1_file.h"
#include "test_controller_hidhid1_file.h"
#include "test_controller_hid_2_file.h"
#include "test_controller_hidhid2_file.h"
#include "test_controller_hid_3_file.h"
#include "test_controller_hid3out_file.h"


// in_size, hid1_size, hid2_size, hid3_size, out_size, type;
// net configs; inhid, hid_1, hidhid1, hid_2, hidhid2, hid_3, hid3out,
NetworkControllerConf_Korneel const conf = {13, 256, 256,128, 4, 1, &conf_inhid, &conf_hid_1, &conf_hidhid1, &conf_hid_2, &conf_hidhid2, &conf_hid_3, &conf_hid3out};