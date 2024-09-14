import os
from mmcl.config_update import (
    change_config_name, update_config, remove_from_config
)

def init_path():
    # path: .../simplex/RecZoo/
    init_path = "<path>/matching/cf/SimpleX/src/__init__.py"
    with open(init_path) as input:
        # Read non-empty lines from input file
        lines = [line for line in input if line.strip()]
    with open(init_path, "w") as output:
        for line in lines:
            output.write(line)
            output.write("\n")
        output.write("from recbox.third_party.daisy.model.LightGCNRecommender import LightGCN\n")

def prepare_config():
    # Copying an old config to new one and updating fields of the new one
    ## Example: Yelp - LGN
    config_path = "<path>/matching/cf/SimpleX/config/LGN_CCL_yelp18_m1/model_config.yaml"
    old_config_name = "MF_CCL_yelp18_m1"
    new_config_name = "LGN_CCL_yelp18_m1"
    change_config_name(config_path, old_config_name=old_config_name, new_config_name=new_config_name)
    field_name_2_new_value = {
        "model" : "LightGCN"
    }
    update_config(config_path=config_path, config_name=new_config_name, field_name_2_new_value=field_name_2_new_value)
    field_names_to_remove = [
        "debug_mode"
    ]
    remove_from_config(config_path=config_path, config_name=new_config_name, field_names=field_names_to_remove)



# init functions
# Path in examples:  path: .../simplex/RecZoo/
prepare_config()
init_path()

# Then run using the run_lgn_simplex.bash