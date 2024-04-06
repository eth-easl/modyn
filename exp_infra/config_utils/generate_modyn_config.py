import yaml
import json
import copy

if __name__ == '__main__':
    template = './modyn/config/examples/modyn_config_template.yaml'
    print(f"Generate modyn config from template {template}")

    with open(template, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets_template = config["storage"]["datasets"]

    train_datasets = []
    for d in datasets_template:
        train_d = copy.deepcopy(d)
        train_d["name"] = d["name"] + '_train'
        train_d["base_path"] = d["base_path"] + '_train'
        train_datasets.append(train_d)
    
    with open('./exp_infra/dataset_metadata.json', 'r') as f:
        dataset_metadata = json.load(f)
    years_per_dataset = {}
    for k, v in dataset_metadata.items():
        years_per_dataset[k] = list(v["total_samples_per_file"].keys())

    test_datasets = []
    for d in datasets_template:
        name = d["name"]
        for y in years_per_dataset[name]:
            test_d = copy.deepcopy(d)
            test_d["name"] = name + '_test_' + y
            test_d["base_path"] = d["base_path"] + '_test/' + y
            test_datasets.append(test_d)

    config["storage"]["datasets"].extend(train_datasets)
    config["storage"]["datasets"].extend(test_datasets)

    output_f = './modyn/config/examples/modyn_config.yaml'
    with open(output_f, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
