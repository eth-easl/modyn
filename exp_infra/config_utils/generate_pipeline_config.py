import yaml
import json
import copy
import pathlib

def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter) # to use with safe_dump

if __name__ == '__main__':
    template_dir = pathlib.Path('./exp_infra/pipeline_configs/templates')
    datasets = ["yearbook", "huffpost", "arxiv"]
    exp_types = ["time", "amount", "datadrift"]

    with open('./exp_infra/dataset_metadata.json', 'r') as f:
        dataset_metadata = json.load(f)
    years_per_dataset = {}
    for k, v in dataset_metadata.items():
        years_per_dataset[k] = list(v["total_samples_per_file"].keys())

    for d in datasets:
        for e in exp_types:
            template = template_dir / d / f"{d}_{e}.yaml"
            print(f"Generate pipeline config from template {template}")

            with open(template, 'r') as f:
                config = yaml.safe_load(f)

            config["data"]["dataset_id"] = f"{d}_train"

            eval_template = config["evaluation"]["datasets"][0]

            eval_datasets = []
            for y in years_per_dataset[d]:
                eval_d = copy.deepcopy(eval_template)
                eval_d["dataset_id"] = f"{d}_test_{y}"
                for m in eval_d["metrics"]:
                    if "evaluation_transformer_function" in m and m["evaluation_transformer_function"][-1] != "\n":
                        m["evaluation_transformer_function"] += "\n"
                eval_datasets.append(eval_d)
            
            config["evaluation"]["datasets"] = eval_datasets
    
            output_f = f'./exp_infra/pipeline_configs/{d}/{d}_{e}.yaml'
            with open(output_f, 'w') as f:
                yaml.safe_dump(config, f, sort_keys=False)