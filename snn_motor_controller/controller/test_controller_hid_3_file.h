#include "../Neuron.h"

// Addition/decay/reset constants as const array here, use pointer in
// configuration struct
float const d_i_hid_3[] = {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f};
float const d_v_hid_3[] = {0.355097f, 0.835244f, 0.892276f, 0.691910f, 0.720800f, 0.056700f, 0.399113f, 0.865410f, 0.555066f, 0.170485f, 0.615753f, 0.831624f, 0.307661f, 0.648354f, 0.425726f, 0.937046f, 0.238516f, 0.170460f, 0.864685f, 0.776071f, 0.294991f, 0.429610f, 0.319439f, 0.954029f, 0.195712f, 0.321771f, 0.784073f, 0.453981f, 0.503656f, 0.162375f, 0.681143f, 0.781633f, 0.962516f, 0.878143f, 0.767533f, 0.919638f, 0.370038f, 0.894883f, 0.637807f, 0.619002f, 0.054456f, 0.292929f, 0.272061f, 0.575298f, 0.100767f, 0.264027f, 0.308055f, 0.193650f, 0.709439f, 0.888976f, 0.866776f, 0.310428f, 0.284188f, 0.787954f, 0.787596f, 0.482420f, 0.645180f, 0.760868f, 0.625125f, 0.285433f, 0.833739f, 0.242481f, 0.016870f, 0.832471f, 0.574514f, 0.642974f, 0.684586f, 0.581266f, 0.237231f, 0.385784f, 0.832160f, 0.169826f, 0.489693f, 0.587009f, 0.319042f, 0.761237f, 0.895139f, 0.840962f, 0.196372f, 0.785236f, 0.073496f, 0.524522f, 0.708123f, 0.401889f, 0.552295f, 0.198667f, 0.345678f, 0.196556f, 0.556238f, 0.554331f, 0.611876f, 0.754170f, 0.526693f, 0.502683f, 0.837215f, 0.924546f, 0.489224f, 0.945597f, 0.778276f, 0.872449f, 0.835273f, 0.098224f, 0.231592f, 0.778715f, 0.585118f, 0.270986f, 0.563036f, 0.979028f, 0.283741f, 0.443115f, 0.957283f, 0.790919f, 0.104680f, 0.271073f, 1.000000f, 0.871156f, 0.686676f, 0.378532f, 0.183557f, 0.046137f, 0.700899f, 0.924369f, 0.436083f, 0.633586f, 0.899120f, 0.766610f, 0.075694f, 0.796666f};
float const t_h_hid_3[] = {0.955965f, 0.316593f, 0.571428f, 0.176401f, 0.275572f, 0.429781f, 0.121264f, 0.721873f, 0.609516f, 0.025583f, 0.374480f, 0.926309f, 0.466726f, 0.872651f, 0.000000f, 0.249265f, 0.221859f, 0.184073f, 0.505281f, 0.794140f, 0.904232f, 0.109388f, 0.775681f, 0.072075f, 0.112925f, 0.206089f, 0.464036f, 0.194701f, 0.037580f, 0.789559f, 0.653619f, 0.061604f, 0.812531f, 0.327565f, 0.089008f, 0.336532f, 0.094148f, 0.080438f, 0.720136f, 0.955583f, 0.103385f, 0.773505f, 0.085510f, 0.845694f, 0.362562f, 0.866759f, 0.279096f, 0.459628f, 0.000000f, 0.202685f, 0.174375f, 0.687651f, 0.785869f, 0.885276f, 0.851665f, 0.293904f, 0.429003f, 0.218103f, 0.516380f, 0.776711f, 0.921201f, 0.211878f, 0.418331f, 0.661262f, 0.704250f, 0.532393f, 0.047626f, 0.650728f, 0.547463f, 0.014620f, 0.755171f, 0.289123f, 0.162449f, 0.577985f, 0.868978f, 0.385948f, 0.604042f, 0.709606f, 0.738191f, 0.400881f, 0.810076f, 0.784114f, 0.000000f, 0.311460f, 0.872384f, 0.675148f, 0.855614f, 0.311429f, 0.599200f, 0.095003f, 0.296701f, 0.106253f, 0.197828f, 0.020696f, 0.926537f, 0.945025f, 0.542598f, 0.492127f, 0.075241f, 0.038172f, 0.233498f, 0.964985f, 0.954486f, 0.182092f, 0.853394f, 0.205283f, 0.401235f, 0.468704f, 0.537253f, 0.081765f, 0.825499f, 0.887523f, 0.938308f, 0.637816f, 0.982332f, 0.865627f, 0.707678f, 0.233735f, 0.759300f, 0.551540f, 0.324982f, 0.234695f, 0.351284f, 0.261081f, 0.442882f, 0.127727f, 0.897730f, 0.495261f};
// size,d_i, d_v, th, v_rest
NeuronConf const conf_hid_3 = {128, 1, d_i_hid_3, d_v_hid_3, t_h_hid_3, 0.0f};