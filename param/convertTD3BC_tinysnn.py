from string import Template
import torch

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_softreset_integrator_from_template, create_connection_from_template_with_weights, create_snntorch_neuron_from_template
from pprint import pprint


def SpikingNet_Tianshou_to_TinySNN(state_dict):
    keys_to_keep = []
    for name in state_dict.keys():
        if name.startswith("actor."):
            keys_to_keep.append(name)
            print(name)

    state_dict = {k: state_dict[k] for k in keys_to_keep}
    # print(state_dict.keys())

    # Create header files
    print("Creating header files")
    print("Size of first layer:",state_dict["actor.preprocess.model.layer_in.weight"].size())
    print("Size of second layer:",state_dict["actor.preprocess.model.layer_out.weight"].size())
    print("Size of mu layer:",state_dict["actor.mu.model.0.weight"].size())
    actor_conf_params = {
        'input_size': state_dict["actor.preprocess.model.layer_in.weight"].size()[1],
        'hidden_size': state_dict["actor.preprocess.model.layer_in.weight"].size()[0],
        'hidden2_size': state_dict["actor.preprocess.model.layer_out.weight"].size()[0],
        'output_size': state_dict["actor.mu.model.0.weight"].size()[0],
        'type': 1,
    }
    pprint(actor_conf_params)
    actor_conf_template = 'param/templates/test_td3_actor_conf.templ'
    actor_conf_out = 'param/controller/test_actor_conf.h'

    create_from_template(actor_conf_template, actor_conf_out, actor_conf_params)
    print("template created!")
    ################### test_actor_inhid_file
    create_connection_from_template('inhid', state_dict, 'actor.preprocess.model.layer_in.weight','actor.preprocess.model.layer_in.bias')

    ################### test_actor_hid_1_file
    create_snntorch_neuron_from_template('hid_1', state_dict, 'actor.preprocess.model.lif_in')



    ################### test_actor_hidhid2_file
    create_connection_from_template('hidhid1', state_dict, 'actor.preprocess.model.layer_out.weight','actor.preprocess.model.layer_out.bias')

     ################### test_actor_hid_3_file
    create_snntorch_neuron_from_template('hid_2', state_dict, 'actor.preprocess.model.lif_out')

    ################### test_actor_hidh3out_file
    create_connection_from_template('hidout', state_dict, 'actor.mu.model.0.weight','actor.mu.model.0.bias')

    print("Done")

if __name__ == "__main__": 
    # Load network
    from torch import nn
    import spikingActorProb
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.mu = nn.Linear(128,4)
            self.sigma = nn.Linear(128,4)
        def forward(self, x):
            x = self.model(x)
            if isinstance(x, tuple):
                x = x[0]    #spiking
            return nn.Tanh()(self.mu(x))
        def reset(self):
            if hasattr(self.model, 'reset'):
                self.model.reset()
  
    state_dict = torch.load(f"TD3BC_TEMP.pth",map_location=torch.device('cpu'))
    # create model
    filtered_dict = {}
    keys_to_keep = []
    for name in state_dict.keys():
        # print(name)
        if name.startswith("actor."):
            keys_to_keep.append(name)
            filtered_dict[name] = state_dict[name]
            # print(name)
        elif name.startswith("model."):
            name_new = name.replace("model.", "actor.preprocess.model.")
            filtered_dict[name_new] = state_dict[name]
            keys_to_keep.append(name)
            # print(name)
        elif name.startswith("preprocess."):
            name_new = name.replace("preprocess.", "actor.preprocess.")
            filtered_dict[name_new] = state_dict[name]
            keys_to_keep.append(name)
            # print(name)
        elif name.startswith("mu."):
            name_new = name.replace("mu.", "actor.mu.model.0.")

            filtered_dict[name_new] = state_dict[name]
            keys_to_keep.append(name)
            # print(name)
    SpikingNet_Tianshou_to_TinySNN(filtered_dict)
 