from options import update_options, options, reset_options

def string_rename(old_string, new_string, start, end):
    new_string = old_string[:start] + new_string + old_string[end:]
    return new_string

def get_bigmodule_list(children):#useless
    Bigmodule = []
    for name, module in children:
        Bigmodule.append(name)
    return Bigmodule
    
def modify_state_dict(pretrained_dict, old_prefix, new_prefix):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if  k.startswith("nn_encoder") and (not options.checkpoint_2d):
            print("Missing key(s) in state_dict :{}".format(k))
        elif options.checkpoint_2d and (not options.checkpoint_3d):
            if k.startswith("nn_encoder"):
                state_dict[k] = v
        else:
            if k not in old_prefix:
            # state_dict.setdefault(k, v)
                state_dict[k] = v
            else:
                for o, n in zip(old_prefix, new_prefix):
                    prefix = k[:len(o)]
                    if prefix == o:
                        kk = string_rename(old_string=k, new_string=n, start=0, end=len(o))
                        print("rename layer modules:{}-->{}".format(k, kk))
                        state_dict[kk] = v
    return state_dict