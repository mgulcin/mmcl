import yaml

def change_config_name(config_path, old_config_name, new_config_name):
    with open(config_path) as f:
         list_doc = yaml.safe_load(f)
            
    print("--------")
    print(list_doc)
    print("--------")
    print(list_doc[old_config_name])
    print("--------")
    
    list_doc[new_config_name] = list_doc[old_config_name]
    list_doc.pop(old_config_name)
    
    with open(config_path, "w") as f:
        yaml.dump(list_doc, f, default_flow_style=False)
    
    print("--------")
    print(list_doc[new_config_name])
    print("--------")
    print(list_doc)
    print("--------")
    
    
    
def update_config(config_path, config_name, field_name_2_new_value):

    with open(config_path) as f:
         list_doc = yaml.safe_load(f)

    print(list_doc)
    print("--------")
    print(list_doc[config_name])
    
    for field_name, new_value in field_name_2_new_value.items():
        if field_name in list_doc[config_name]:
            print(f'prev {field_name}: {list_doc[config_name][field_name]}')
        list_doc[config_name][field_name] = new_value
        print(f'after {field_name}: {list_doc[config_name][field_name]}')

    with open(config_path, "w") as f:
        yaml.dump(list_doc, f, default_flow_style=False)

    print("--------")
    print(list_doc[config_name])


def remove_from_config(config_path, config_name, field_names):
    with open(config_path) as f:
         list_doc = yaml.safe_load(f)

    print(list_doc)
    print("--------")
    print(list_doc[config_name])
    
    for field_name in field_names:
        if field_name in list_doc:
             list_doc.pop(field_name)
        elif field_name in list_doc[config_name]:
            list_doc[config_name].pop(field_name)

    with open(config_path, "w") as f:
        yaml.dump(list_doc, f, default_flow_style=False)

    print("--------")
    print(list_doc[config_name])