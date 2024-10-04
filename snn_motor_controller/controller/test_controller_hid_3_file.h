#include "../Neuron.h"

// Addition/decay/reset constants as const array here, use pointer in
// configuration struct
float const d_i_hid_3[] = {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f};
float const d_v_hid_3[] = {0.623491f, 1.000000f, 0.575941f, 0.569374f, 1.000000f, 0.787774f, 0.863888f, 0.744158f, 0.759440f, 0.842956f, 0.787622f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.858368f, 1.000000f, 1.000000f, 0.824543f, 1.000000f, 1.000000f, 1.000000f, 0.837874f, 1.000000f, 1.000000f, 0.532084f, 1.000000f, 0.865225f, 0.671678f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.719017f, 1.000000f, 0.669781f, 1.000000f, 0.753874f, 1.000000f, 0.517371f, 1.000000f, 1.000000f, 1.000000f, 0.521987f, 0.616250f, 1.000000f, 0.533983f, 1.000000f, 0.766105f, 0.745420f, 0.608721f, 1.000000f, 0.644376f, 0.704465f, 0.000000f, 1.000000f, 1.000000f, 0.722718f, 0.693329f, 1.000000f, 1.000000f, 1.000000f, 0.691500f, 1.000000f, 1.000000f, 0.742837f, 1.000000f, 1.000000f, 0.594358f, 1.000000f, 1.000000f, 0.621811f, 1.000000f, 0.658787f, 0.783431f, 0.703646f, 0.634919f, 0.453752f, 0.686230f, 0.848797f, 1.000000f, 0.875347f, 1.000000f, 0.725060f, 0.739729f, 0.737229f, 0.661500f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.813594f, 1.000000f, 1.000000f, 0.729858f, 1.000000f, 0.866312f, 1.000000f, 1.000000f, 1.000000f, 0.518829f, 1.000000f, 0.708406f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.560411f, 0.924747f, 1.000000f, 0.839864f, 1.000000f, 0.000000f, 0.712969f, 1.000000f, 0.946490f, 0.796584f, 1.000000f, 0.640019f, 0.585280f, 0.819935f, 1.000000f, 0.607932f, 1.000000f, 0.731254f, 1.000000f};
float const t_h_hid_3[] = {3.839889f, 5.372096f, 3.386710f, 3.471422f, 5.691617f, 4.260698f, 4.441048f, 4.376073f, 4.173261f, 4.804592f, 4.800403f, 5.486126f, 4.992507f, 4.597992f, 5.695975f, 4.314666f, 5.549253f, 5.643701f, 3.893042f, 5.183526f, 5.231950f, 5.161345f, 3.761496f, 5.638950f, 5.167311f, 3.837571f, 5.383282f, 4.621377f, 4.027909f, 4.359928f, 4.653484f, 5.718245f, 4.833097f, 5.368291f, 4.348586f, 4.582717f, 3.587058f, 5.024287f, 3.688989f, 5.252971f, 4.383976f, 5.835348f, 5.224204f, 5.080032f, 3.896288f, 3.947038f, 5.298971f, 3.768659f, 4.757403f, 4.439631f, 4.439312f, 3.828831f, 5.241602f, 3.895261f, 4.659701f, 0.307471f, 4.582065f, 4.697758f, 2.511843f, 3.961222f, 4.510604f, 5.678492f, 5.572164f, 4.504879f, 5.187559f, 5.469814f, 4.755333f, 5.939505f, 6.337888f, 3.461418f, 4.680104f, 4.202124f, 3.071667f, 4.961322f, 3.935208f, 3.995652f, 3.844631f, 4.365214f, 3.764646f, 4.149783f, 4.487478f, 5.210356f, 3.711885f, 5.188609f, 4.355433f, 4.003266f, 3.376669f, 3.351548f, 5.332115f, 4.114771f, 4.714268f, 4.926589f, 4.343729f, 4.146427f, 5.708584f, 4.410951f, 5.397687f, 4.782506f, 4.874411f, 4.987525f, 5.153279f, 3.807401f, 5.573159f, 3.706068f, 5.163823f, 4.967425f, 5.114299f, 5.140081f, 5.416251f, 3.277941f, 4.641167f, 5.164769f, 4.220928f, 5.122685f, 0.378061f, 4.261361f, 4.898066f, 3.734757f, 4.348226f, 4.862329f, 3.071965f, 3.691362f, 4.309751f, 5.199724f, 3.859375f, 4.108064f, 3.975688f, 4.269807f};
// size,d_i, d_v, th, v_rest
NeuronConf const conf_hid_3 = {128, 1, d_i_hid_3, d_v_hid_3, t_h_hid_3, 0.0f};