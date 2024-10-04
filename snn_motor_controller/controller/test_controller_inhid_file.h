#include "../functional.h"
#include "../Connection.h"
  
// Weights and bias as const array here, use pointer in configuration struct
float const w_inhid[] = {-1.367667f, 2.241009f, -2.507279f, -0.442594f, 1.139071f, -0.573368f, 1.806782f, 0.324989f, -0.037997f, -2.753857f, -0.498834f, -0.179057f, -0.225891f, -0.609411f, -2.628853f, -1.202519f, 0.236561f, -1.133828f, -1.045304f, 1.644213f, -0.091222f, 5.782222f, -1.390852f, 0.033939f, -0.363601f, -0.429558f, 0.829580f, 0.219167f, -0.023500f, 0.351864f, 0.470240f, -0.323097f, -0.032328f, -4.042367f, -3.119649f, -6.955173f, -0.146348f, 0.178645f, 0.036909f, 2.667106f, 1.989829f, -0.375547f, 0.603553f, 0.117524f, 0.195762f, 1.391283f, 0.192661f, 0.380496f, 1.138750f, -0.237773f, -0.532256f, 0.614294f, -1.645126f, 0.503594f, -1.565781f, -0.976103f, -0.416860f, -0.278538f, -1.390681f, -2.163734f, -1.660831f, -0.964474f, 0.304777f, 0.832352f, -0.324289f, 0.248584f, 1.149226f, 1.639434f, -0.085079f, 0.380601f, -0.007078f, 0.911839f, -0.832126f, 0.976131f, 2.391691f, 0.289150f, -0.324568f, -2.131289f, -1.483017f, 1.338060f, -1.092521f, 0.029426f, 1.052241f, -1.160430f, 1.408390f, -3.148931f, -4.778560f, -2.522459f, 0.076525f, 0.009504f, 0.654637f, 2.203324f, 1.218065f, 1.731506f, 0.074095f, 0.109772f, 1.327391f, 0.191095f, -1.945868f, -4.032728f, -0.447371f, 0.695495f, 0.201316f, -0.258693f, 0.043087f, -0.185777f, 0.023946f, 0.027256f, -0.354934f, -0.040675f, -5.276748f, 5.293069f, 0.153804f, 0.857931f, 0.226084f, -0.005549f, -0.059057f, -0.857861f, 1.178156f, 0.232448f, -0.561564f, 0.559401f, -0.202811f, 2.556892f, -2.933193f, -5.152275f, -3.858359f, 0.057126f, -0.248450f, -0.482781f, -2.021950f, 0.842115f, -0.907788f, -0.468165f, 1.375569f, -0.447865f, -1.767480f, -4.125871f, -1.718076f, -0.340404f, 0.447884f, 0.314844f, -0.208061f, 1.140798f, 0.103491f, -0.824731f, 0.465923f, -0.010241f, -0.393837f, -5.481755f, -2.552397f, -3.530985f, 1.696259f, 0.195659f, -0.037062f, -0.432352f, -0.363270f, 0.077867f, 4.058908f, -0.290355f, 0.396248f, 0.581290f, -0.950346f, -0.670962f, 2.238421f, -5.760964f, -0.057802f, -0.076149f, -0.604620f, 0.582247f, 1.099590f, -0.121850f, -0.319302f, 0.409758f, 0.265174f, 1.216235f, -0.247465f, 1.947651f, 6.840220f, -0.080633f, -0.173315f, 0.518093f, -1.186034f, 0.306535f, 1.107309f, -0.478193f, 0.542881f, -0.150501f, 2.003928f, 3.236228f, -0.407216f, 6.302929f, -0.073752f, 0.203900f, -0.077665f, 2.337178f, 0.138416f, -0.458366f, 0.961934f, 0.171169f, 0.165969f, 1.502225f, 1.989118f, 1.187700f, 0.270757f, 0.470961f, -1.503068f, 0.292362f, 1.160622f, 0.815920f, -0.142882f, 0.577022f, -0.257093f, 0.161564f, -2.530822f, 3.239711f, 1.622481f, -6.004553f, 0.261040f, -0.228797f, 0.302589f, -1.747265f, 1.334863f, -0.074107f, -0.631671f, 0.793103f, 0.124245f, -3.095370f, 2.859245f, -1.044800f, -3.754044f, -0.114924f, 0.396144f, 0.109808f, -1.755390f, 1.017831f, 0.734844f, -0.590592f, 0.014291f, 0.649311f, 2.816304f, 1.876463f, -3.154955f, 5.209738f, 0.295695f, -0.181725f, 0.049766f, -1.458319f, 1.044190f, -0.453582f, -0.783373f, -0.253118f, 0.447477f, 2.416730f, 1.863934f, -2.132261f, -5.869219f, -0.163885f, 0.192022f, 0.023091f, 0.280357f, 0.512416f, 2.541962f, 0.460409f, 0.857001f, 0.707978f, 2.355015f, -0.900405f, 2.092449f, 4.740546f, 0.143413f, 0.107042f, 0.087182f, 0.833813f, -1.973913f, -0.626921f, 0.558270f, -0.537644f, -0.227527f, -2.776030f, -4.979551f, -0.160618f, 0.445067f, -0.222431f, -0.028157f, 0.074956f, -1.549844f, -0.680852f, -0.181610f, -0.265627f, -0.597132f, -0.156456f, 1.545297f, -0.548324f, 3.407396f, 3.553672f, -0.157720f, 0.046626f, 0.978257f, -2.646030f, -0.267597f, 1.535150f, -0.541658f, 0.227084f, -0.602433f, 2.709715f, 1.444103f, -1.501675f, 6.664567f, 0.256548f, -0.023071f, 0.233052f, 1.445302f, -0.032714f, 0.711292f, 0.415205f, -0.856182f, 0.200527f, -0.633119f, 5.368256f, -2.347002f, -5.246017f, -0.039925f, 0.138515f, 0.637103f, 1.130250f, -0.674998f, -1.074038f, 0.109685f, -0.640387f, -0.690267f, -3.975852f, 4.115973f, 0.843173f, -0.792144f, 0.136310f, 0.011158f, 0.263150f, -1.926648f, 0.656018f, -0.976303f, -0.905704f, 0.115383f, 0.436577f, -0.475976f, 0.178373f, 5.418431f, -1.109042f, 0.058317f, 0.227053f, 0.617529f, 0.251860f, -0.827778f, -1.011783f, 0.325430f, 0.057567f, -0.520285f, -0.384936f, -1.061078f, -2.275429f, 0.272388f, -0.542371f, -0.335193f, -0.188423f, -1.112394f, 0.129741f, 0.651370f, -0.309969f, -0.018412f, 0.271929f, 0.097467f, 0.193959f, 3.406628f, 1.922400f, -0.390783f, -0.029691f, 0.492598f, 1.601948f, -0.377788f, 1.531410f, 0.422640f, 0.246649f, 0.371398f, 2.806425f, -4.006203f, 5.999609f, 0.038845f, -0.184203f, 0.337604f, -0.461109f, -0.398062f, -0.763637f, 0.158541f, -0.109408f, -0.387539f, 0.062197f, 0.257952f, 6.865015f, -3.006818f, 1.383504f, 0.251060f, -0.241522f, 0.231940f, 0.485094f, 3.326364f, -0.746030f, 0.295267f, 0.932475f, -0.471553f, -3.390043f, -1.901217f, 2.639199f, 3.572332f, 0.012417f, 0.022285f, 0.121281f, 0.371573f, -1.463771f, -2.204883f, 0.403050f, -0.241782f, -0.620251f, 1.334332f, 6.788567f, 1.182284f, 1.286138f, -0.247547f, -0.197133f, 0.523995f, -0.022899f, -2.287342f, -2.309980f, 0.401613f, -0.919093f, -0.206640f, 1.620972f, 1.794251f, -0.076790f, -0.855419f, -0.677360f, 0.281187f, 0.229940f, -1.963079f, 0.451388f, -0.521069f, -0.242146f, 0.173849f, -1.071610f, 0.068408f, 2.857362f, 3.100173f, -2.609575f, 0.486753f, 0.176115f, 0.427431f, -1.848496f, 0.335692f, -0.388085f, -0.721725f, 0.099085f, 0.248729f, -1.201579f, 1.377635f, -4.080031f, 1.603258f, 0.103397f, 0.057545f, -0.164168f, 0.325761f, -1.755564f, 3.620378f, -0.112791f, 0.387557f, 0.325537f, -0.595754f, -0.433522f, -0.927071f, -2.722070f, -0.180221f, 0.050030f, -2.486422f, 0.146775f, 0.479211f, 2.628490f, 0.986351f, 0.598644f, 0.834965f, -2.997361f, 5.084718f, -0.884113f, 0.293649f, 0.338843f, 0.293996f, 0.139014f, 4.109516f, 0.350799f, 1.506864f, -0.164337f, 1.560075f, 1.162882f, 0.455954f, 1.681803f, -1.618832f, 0.946936f, 0.088246f, -0.451301f, -0.045702f, -2.989291f, 0.579848f, -2.069729f, 0.004718f, 0.468403f, -0.795552f, -4.192215f, -1.466745f, 1.152996f, -3.484238f, -0.065609f, 0.177510f, 1.472519f, 0.716905f, 0.032561f, -1.713364f, -0.180669f, 0.132862f, 0.134429f, 1.345777f, -2.993806f, 0.034514f, -1.965469f, 0.409808f, 0.533845f, 0.566415f, 1.018187f, -2.434875f, 0.915880f, 0.123318f, -0.804915f, 1.269766f, -0.447149f, 2.272952f, 1.375541f, -0.365600f, 0.381787f, 0.146054f, 0.574071f, 1.908214f, -1.113376f, -1.382916f, 0.917706f, -0.157373f, -0.350334f, 0.499103f, 5.235659f, 0.609805f, 5.155527f, 0.209904f, -0.048042f, 0.529382f, -1.667602f, 0.994684f, -1.004736f, -0.559997f, 0.314396f, -0.242312f, 2.261859f, -0.111315f, -3.988162f, -3.118289f, 0.292973f, 0.239110f, 0.451524f, 0.055415f, -2.880170f, 0.488664f, -0.381030f, -0.264072f, -0.514061f, -2.663753f, -0.283677f, 3.298365f, 0.802728f, -0.365286f, 0.476898f, -0.760509f, -0.420728f, 1.607277f, -0.758739f, -0.046124f, 0.501265f, -0.171242f, -1.635411f, -5.161053f, 1.187460f, 2.512405f, 0.162983f, -0.033614f, 0.223120f, -0.006093f, -0.112507f, 0.411833f, -0.439251f, 0.043338f, -0.236862f, -5.105674f, 0.746647f, 1.793611f, 3.106417f, -0.161318f, 0.239748f, -0.340778f, -0.573128f, 0.317107f, 1.706065f, 0.013311f, -0.085038f, 0.564844f, 4.621730f, 2.262397f, 0.010374f, -0.773886f, 0.183975f, 0.149617f, -0.434386f, 1.340776f, -0.217980f, 0.280315f, 0.654604f, -0.065286f, -0.114199f, 4.575560f, -2.244576f, 1.398393f, 3.106897f, -0.315253f, -0.006681f, 0.108863f, 0.181085f, 1.516430f, 1.062134f, -0.135301f, 0.985483f, 0.296553f, -0.080163f, -0.827474f, -1.005915f, 1.478638f, 0.939674f, -0.866877f, 0.271926f, 2.832956f, -1.584682f, -0.055610f, 0.516598f, -0.695016f, -0.130021f, 3.080251f, 2.615432f, -2.007701f, -4.512504f, 0.074671f, -0.010218f, -0.983295f, 0.102551f, -1.330878f, 2.034198f, 0.228700f, -0.190145f, 0.758543f, -0.406843f, 2.014954f, 5.292327f, 1.473315f, -0.160340f, 0.033928f, -0.323730f, 0.098750f, 0.518608f, 0.585037f, 0.418336f, 0.289675f, -0.322778f, -2.912784f, 2.643170f, 1.745294f, -1.727594f, -0.318205f, -0.840725f, -0.879853f, 0.554625f, -0.345146f, 1.656826f, 0.435010f, 0.111670f, -0.006998f, -0.498101f, 3.353597f, -7.100177f, 0.099726f, -0.206714f, 0.088208f, -0.084122f, 0.355568f, 0.370143f, -2.162794f, 0.130874f, 0.366813f, -0.731469f, -0.007673f, 1.886523f, -1.186567f, -1.077950f, -0.187955f, -0.310003f, -0.240853f, 3.461906f, 0.672459f, 2.894647f, 1.318164f, 0.673956f, 0.321747f, -0.663949f, -1.801300f, 0.365708f, 0.782190f, -0.083646f, 0.240957f, 0.174067f, 0.047740f, 0.686966f, 2.305733f, -0.217751f, 0.292189f, 1.219847f, -0.340236f, 2.557298f, -4.214608f, 0.633122f, 0.036601f, -0.451289f, 0.034617f, 1.200668f, -0.688216f, 0.923884f, 1.069744f, 0.786200f, 0.628687f, 0.714503f, -0.613748f, -0.185172f, -3.639576f, -0.864739f, -0.009468f, -0.315332f, -0.566466f, 0.226172f, -1.282018f, -0.022450f, 0.252870f, -0.419419f, 1.295354f, 1.288419f, -2.700172f, -0.312332f, 0.650322f, -0.252366f, -0.817085f, -2.315978f, -0.479312f, 2.222975f, -0.795347f, -0.950187f, 1.013394f, 0.773340f, -1.813684f, -4.709000f, -1.140476f, 0.033303f, -0.044515f, -0.210843f, 1.689578f, -0.678433f, 0.941476f, 0.486867f, -0.373907f, 0.475467f, -6.231046f, -3.199454f, -0.586878f, 0.855793f, -0.031714f, -0.190590f, 0.065304f, 0.957830f, -0.826149f, -0.944085f, 0.027958f, -0.304122f, 0.080004f, 1.862960f, 3.111022f, -5.490424f, 4.497408f, -0.160158f, 0.130499f, -0.034234f, 0.185820f, -0.625544f, 0.434482f, -0.318975f, -0.279352f, 0.256940f, -3.391223f, -1.455184f, -1.291477f, -0.410711f, -0.141353f, 0.690635f, -0.320817f, 0.145419f, 1.187801f, 1.128670f, -0.176509f, 0.565444f, 0.607638f, -4.217491f, 4.795496f, -3.621988f, 0.672721f, 0.020398f, 0.333102f, 0.016136f, 2.132672f, 0.951825f, -1.450535f, 0.763896f, 0.417741f, -0.058913f, -1.412184f, -0.032866f, -4.472418f, -4.491842f, -0.145847f, -0.079782f, 0.225580f, -1.820897f, -1.002206f, 0.486758f, -1.257618f, -0.087820f, 1.159130f, 0.457514f, -2.744898f, 4.255204f, 0.072614f, -0.487008f, 0.021959f, -0.847362f, 1.758854f, -0.039355f, -0.452498f, 0.526526f, 0.098091f, 0.137814f, 3.739011f, 2.204345f, 5.167307f, 0.237075f, 0.421460f, 0.283187f, 0.171177f, -0.323681f, -0.353283f, 1.302454f, -0.299214f, 0.408703f, 0.162608f, 0.689718f, 1.504217f, 3.094380f, 1.571075f, -0.671229f, -0.526786f, -0.712387f, -2.174680f, 3.793746f, 1.365174f, -0.283779f, 2.660136f, 0.265976f, -1.376038f, -0.123298f, 0.609884f, 0.194509f, 0.020238f, 0.222352f, -0.181933f, -1.247535f, 1.097964f, -0.112705f, -0.543984f, 0.382870f, 0.011317f, 2.258688f, 1.803600f, 5.542927f, -4.553046f, -0.147055f, -0.131542f, 0.206589f, 2.533641f, 0.524965f, -2.223474f, 0.515251f, 0.459034f, -0.113370f, -3.276593f, 1.375590f, 1.246721f, 2.322475f, 0.304840f, 0.132291f, -0.311347f, 0.543735f, -1.548258f, -0.239021f, 0.150166f, -0.544514f, -0.195579f, 1.646012f, 6.340421f, 2.335897f, 0.067127f, 0.195795f, 0.208545f, -0.001481f, -2.772335f, 1.236803f, 1.643877f, -1.484144f, 0.563874f, 0.227646f, -2.979331f, 1.009719f, -2.265172f, 1.772135f, -0.266036f, -0.264461f, 0.811164f, -0.412698f, 0.541709f, 3.370059f, -0.437710f, -0.059589f, 0.579958f, 2.783583f, 4.670445f, -3.564788f, -2.401419f, 0.175054f, -0.089241f, 0.286793f, -0.259968f, 1.384238f, 0.674041f, -0.530313f, 0.513261f, 0.610537f, -0.491685f, -0.969049f, 0.584489f, -1.883741f, 0.921862f, 0.932792f, 0.519750f, -0.730176f, -2.009290f, -2.380522f, 0.011868f, -0.537351f, 0.052135f, -3.472344f, 0.420314f, -0.236623f, 0.808366f, -0.105174f, -0.526138f, -2.000985f, 1.432812f, 0.967487f, 1.540662f, 0.351714f, 0.398408f, 0.565193f, -3.935630f, 5.715603f, -2.526726f, -1.580705f, -0.243770f, -0.039622f, 0.305340f, -1.089502f, -0.642054f, -0.517799f, -0.096800f, -0.681155f, -0.248002f, 1.151973f, -6.374924f, -3.350118f, 2.147562f, 0.356583f, 0.048325f, -0.784084f, 0.397246f, -0.359246f, 0.158550f, 0.526122f, -0.637082f, -0.349978f, 0.133196f, -2.251124f, -1.654611f, 7.072844f, 0.061128f, -0.138784f, -1.217349f, 1.102770f, -1.440923f, -1.209041f, 0.148694f, 0.232835f, -0.604369f, -1.401960f, -4.286287f, 4.715898f, -3.971381f, 0.097544f, -0.165686f, 0.410367f, -0.559247f, 2.357264f, -0.780509f, -0.158098f, 0.493398f, -0.638596f, -1.046226f, 0.058373f, 1.219315f, 5.539059f, -0.296455f, 0.068347f, 0.461483f, 1.728912f, -1.054148f, -0.988093f, 0.749543f, -0.285485f, -0.061986f, 1.705007f, 0.718673f, 5.171744f, -5.517746f, -0.025830f, -0.108704f, -0.104198f, 1.319357f, -1.499418f, -1.601853f, 0.456718f, 0.085908f, -0.564438f, 2.879962f, -1.577757f, 2.421145f, -2.786588f, -0.455837f, 0.050412f, -0.582425f, -2.136371f, 1.179387f, -1.188003f, 0.243290f, -0.273201f, 0.358497f, 1.873473f, 0.680825f, -0.514879f, -3.390146f, -0.215202f, -0.510033f, 0.838741f, 0.480873f, 0.407471f, 0.162811f, -0.467877f, 0.154817f, 0.081569f, 3.902094f, -5.029791f, 0.741539f, -1.694782f, 0.152465f, -0.190803f, 0.203336f, 1.047544f, 0.104019f, -1.367607f, 0.436236f, 0.131855f, -0.764331f, 0.924307f, -4.669843f, 5.574013f, 2.376153f, -0.187180f, 0.314497f, 0.024555f, -2.175109f, -2.263398f, 1.042571f, -1.820932f, -0.882685f, 0.223091f, -1.626857f, -1.506762f, 1.603061f, -4.020926f, -0.251406f, -0.001028f, -0.062330f, 0.439761f, -3.619301f, 0.403837f, -0.843963f, -1.393998f, -1.091914f, -0.372435f, 0.503060f, -3.568751f, 2.572512f, 0.235939f, -0.288634f, -0.538133f, -0.058547f, 0.265552f, -1.379806f, -0.452515f, 0.350350f, -0.399699f, 3.681365f, -0.050665f, -4.356317f, -0.429281f, -0.326926f, -0.026558f, 0.586810f, 0.149289f, -0.861195f, -0.859856f, 0.263366f, -0.278446f, 0.118915f, -1.334085f, 1.822082f, -1.203088f, 2.471873f, -0.105769f, -0.552207f, 0.268252f, -0.246267f, 2.174156f, -0.152706f, -0.011151f, 0.861907f, 0.261810f, 1.123368f, 6.772069f, -1.932276f, -4.195201f, -0.137425f, 0.043661f, 0.486938f, -1.616424f, 1.727617f, 1.215211f, -0.252243f, 0.594626f, 0.220053f, -0.898303f, 2.343237f, 4.809028f, -2.901629f, -0.129081f, 0.292271f, -0.744539f, 0.419051f, 0.240568f, -0.552979f, -1.284379f, 0.255617f, 1.629442f, 1.614497f, 1.848824f, -2.084114f, -1.692201f, -0.255549f, 0.666467f, 0.233289f, -1.127273f, -1.087281f, -1.261438f, -0.294963f, -0.837141f, -1.719242f, 0.842119f, -1.083583f, -1.235487f, 1.565720f, -0.032882f, 0.699598f, -0.098821f, -1.589707f, -3.709338f, 3.622749f, -0.138094f, -0.506826f, 0.803374f, 1.352631f, 0.596962f, 2.055736f, 2.713402f, -0.075593f, 0.198869f, 2.010223f, -0.400008f, -0.845411f, -2.256378f, 0.127784f, -0.178064f, -0.060257f, 1.542639f, -0.745389f, -5.921797f, -0.469712f, -0.181366f, 0.066306f, -0.576986f, 1.092286f, 0.410731f, 0.993544f, 0.663153f, 0.305380f, -0.366062f, 2.657541f, 1.763676f, 3.881898f, -5.967943f, 0.072180f, -0.115469f, -0.290073f, -0.198057f, 3.025885f, 0.333471f, 1.006736f, 0.651148f, 1.870776f, -0.913981f, -2.584347f, -0.497707f, 0.035504f, 0.057760f, -0.074044f, -0.697753f, -1.199159f, 0.344312f, 0.672458f, -0.710818f, 0.181477f, 0.426392f, -3.161141f, 0.273414f, 7.029606f, -3.302158f, 0.088196f, -0.249883f, 0.039760f, 1.979476f, 1.343395f, -2.479659f, 1.041413f, -0.113665f, -0.415833f, 0.779822f, -2.703354f, 0.955064f, -1.195725f, 0.154556f, 0.215070f, 0.598681f, 0.197864f, 1.792507f, 2.089906f, 0.198055f, 0.322434f, 0.435595f, -1.181022f, -3.706941f, -2.770512f, -5.323184f, 0.373506f, -0.045804f, 0.251084f, 2.533621f, -1.175799f, 0.451547f, 0.759959f, -1.568473f, 0.713134f, -0.427972f, -1.837725f, -0.959363f, -1.373569f, 0.530678f, -0.902152f, -0.017464f, 0.896185f, 0.743727f, 1.330630f, 0.894175f, 0.964255f, 1.311293f, -1.326891f, -3.202202f, -4.108933f, 2.470524f, 0.218893f, -0.018052f, 0.371749f, 1.389116f, 1.406375f, 1.067884f, 0.453754f, 0.270511f, -0.370605f, -1.808542f, 1.276905f, -4.740306f, 4.411705f, -0.028052f, 0.075681f, -0.108510f, 0.338913f, -1.150633f, 0.207224f, 0.235533f, -0.734006f, -0.071463f, 0.112520f, 0.096880f, -0.304375f, -2.084549f, -0.153743f, 0.462341f, -2.418784f, -0.375279f, 0.234247f, -0.830394f, -0.659224f, -0.030612f, 0.333237f, -0.409963f, -0.946864f, 6.796033f, 3.924644f, 0.079597f, -0.303378f, -0.155868f, -0.530953f, 0.977246f, 0.905175f, -0.254707f, 0.603130f, -0.155149f, 0.020487f, 8.234630f, -1.770286f, -2.041188f, -0.275676f, 0.221945f, 0.113447f, 2.469467f, -0.498764f, 0.599218f, 1.090031f, -0.423841f, -0.281066f, -1.574071f, -0.846380f, 0.128753f, 1.583876f, -0.930692f, -0.240755f, 0.677185f, -1.059306f, 0.507867f, 0.948459f, -0.425030f, -0.002678f, 0.023475f, -2.987993f, 0.162233f, 1.984593f, 3.497383f, 0.436213f, 0.285730f, -0.294589f, -0.577677f, -0.472034f, 1.014625f, -0.361360f, -0.264186f, 0.402178f, 2.473562f, 2.415298f, -6.626879f, 0.914579f, 0.102717f, -0.387015f, -0.275336f, -1.115592f, -3.078763f, 2.440018f, -0.179448f, -1.041658f, 0.872103f, -1.677724f, -3.068039f, -2.430756f, 2.573862f, -0.077626f, -0.124549f, 0.205686f, 0.610664f, -0.403834f, -0.299368f, 0.408198f, 0.042603f, -0.400265f, -1.688347f, 4.967577f, 6.247454f, 1.296473f, 0.237856f, 0.172081f, 0.267780f, 0.743959f, 1.078711f, 0.177526f, 0.364418f, 0.883965f, 0.007595f, -2.068794f, -1.546953f, -5.383327f, 0.419388f, 0.170490f, 0.125440f, -0.132451f, 0.803656f, 0.388609f, 3.445024f, 0.292386f, -0.029806f, 0.551307f, 0.061313f, -8.331705f, -0.220727f, 1.012202f, -0.077344f, 0.087312f, -0.173568f, -0.980427f, -0.434339f, 0.503861f, 2.259303f, -0.888975f, 0.134630f, 1.843946f, 1.924946f, 0.326849f, 1.058016f, 0.085351f, -0.427098f, 0.187583f, 0.762360f, 3.121267f, 0.274774f, 0.406111f, 1.087672f, -0.024737f, -1.586840f, -3.433771f, -1.372306f, 4.975385f, -0.070502f, -0.163638f, 0.025021f, -0.373672f, 0.589049f, 0.024459f, 0.007875f, 0.274540f, -0.269598f, -0.730323f, -6.441890f, -1.018759f, 3.656259f, -0.297664f, 0.033517f, 0.195645f, 0.608538f, -0.002366f, -1.218429f, 0.110970f, 0.136675f, -1.216405f, -1.587115f, 3.130510f, -5.135967f, 2.648728f, 0.137681f, 0.254330f, -0.428231f, 0.607539f, -0.893842f, 0.464196f, 0.412553f, -0.271854f, -0.038248f, 1.956765f, 2.358901f, 6.548398f, -2.837382f, 0.166557f, 0.255918f, 0.087493f, -1.999742f, -0.027594f, -0.432443f, -0.766019f, 0.144745f, 0.146273f, -0.589542f, 3.678176f, 2.658207f, -3.175012f, -0.524559f, -0.008627f, -0.517908f, -0.323652f, 2.006827f, -1.639772f, 0.480568f, 0.530362f, -0.428976f, -1.997514f, 3.211905f, 3.439209f, 4.546100f, -0.305336f, 0.050892f, -0.706163f, 1.020950f, 0.792660f, 4.579688f, 0.294318f, -0.488015f, 0.890800f, 2.411236f, 0.862592f, -0.811638f, 3.949930f, -0.072682f, -0.009126f, 0.582874f, -0.375428f, -1.058287f, 1.041179f, 0.495668f, -0.622822f, 0.366219f, -4.217854f, -8.185355f, -1.009373f, -1.070284f, 0.349216f, -0.021449f, -0.046823f, -0.252749f, -1.764742f, 1.292657f, -0.312451f, -0.904861f, 0.604549f, -1.241503f, -2.470009f, -2.887401f, -4.639816f, 0.127163f, 0.012073f, -0.112612f, -2.047723f, 0.150808f, 0.196545f, -0.141694f, 0.357373f, 0.681882f, -2.976802f, -0.931885f, 0.232486f, 2.845604f, 0.121132f, 0.267122f, 1.670662f, 0.007776f, 0.827194f, -0.668226f, -0.180299f, -0.085155f, -0.465151f, -4.678015f, -2.055670f, -2.140729f, -4.023902f, -0.230756f, -0.078693f, -0.358085f, -1.305667f, -1.967913f, -1.960338f, -0.616667f, -0.686654f, -1.501366f, -0.461874f, -2.550817f, 0.103589f, -1.912438f, -0.088129f, 0.113349f, 0.855329f, -0.470465f, 0.310254f, 0.364810f, -0.445204f, 0.236948f, 0.140603f, -0.778448f, 4.454683f, 5.596046f, -1.070282f, -0.221787f, -0.197192f, -0.031944f};
float const b_inhid[] = {0.603912f, 0.148962f, 0.434318f, -3.188282f, 1.298180f, -2.491014f, 1.390460f, 0.560576f, 2.239415f, 0.616330f, 1.692565f, -0.010735f, -2.647090f, -0.578826f, -2.619680f, -0.258765f, 0.759400f, -1.478578f, -0.982261f, -1.384200f, -1.346702f, -1.375244f, 3.158067f, -1.090831f, -0.824638f, -1.774232f, 0.477774f, -3.058876f, -3.272708f, 1.156743f, 2.063362f, -1.669732f, -0.055476f, 1.361142f, 1.995603f, -2.463815f, 1.603355f, 0.511680f, 1.661811f, 0.258546f, -2.840702f, -2.308454f, 0.732044f, -2.186232f, -1.837088f, -2.180798f, 2.265030f, -1.373623f, 1.024233f, 1.638536f, -1.200354f, -0.789006f, -0.323176f, 0.791098f, 3.256118f, 2.467630f, -0.771819f, 0.910557f, -2.908421f, 1.830795f, 0.623756f, -0.205065f, -1.826925f, -0.179480f, -1.737214f, 0.351028f, -1.638012f, 0.233807f, 1.071670f, -1.204878f, -1.108171f, -0.748216f, -0.446268f, -1.639488f, 1.257766f, -0.969990f, 0.254042f, 0.213488f, -0.121208f, -0.444541f, 2.297213f, -0.061175f, -2.870094f, -2.190372f, -0.664184f, -0.606551f, 2.339055f, 1.530022f, -1.000376f, -4.152833f, -0.744858f, 2.306134f, 1.239586f, 1.625381f, 0.608400f, 2.153581f, 0.980747f, 0.684598f, -0.912436f, -0.992497f, 0.187458f, 1.447916f, 0.587057f, 2.890029f, -1.411872f, 1.051759f, 0.736063f, 0.791735f, -2.613646f, 0.823149f, -0.878499f, 1.723133f, 2.186641f, -0.460833f, 0.492898f, -0.037321f, 2.219087f, 0.663519f, 0.709337f, 3.269736f, -0.693292f, -1.289797f, -0.838243f, -2.879605f, -1.869035f, -0.555319f, -0.807448f, 1.525021f};

// post, pre, w
ConnectionConf const conf_inhid = {13, 128, w_inhid, b_inhid};
