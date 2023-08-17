import yaml
import urllib.parse
import requests
import json
import time
import pickle

BASE_PIPELINE_FILE = "./pipelines/yearbook_base.yaml"
EXPERIMENTS_FILE = "./pipelines/experiments.json"
CHAT_ID_TELEGRAM = "881642944"
BOT_TOKEN = '6453968346:AAG5iFpKoBSRbVhwlqriiGF6I-ZCG40Fq80'

def create_yaml_experiment_file(experiment, experiments):
    with open(BASE_PIPELINE_FILE, "r") as f:
        base_pipeline = yaml.safe_load(f)
    if "presampling_config" in experiments[experiment]:
        base_pipeline["training"]["selection_strategy"]["config"]["presampling_config"] = experiments[experiment]["presampling_config"]
    if "downsampling_config" in experiments[experiment]:
        base_pipeline["training"]["selection_strategy"]["config"]["downsampling_config"] = experiments[experiment]["downsampling_config"]
    if "epochs_per_trigger" in experiments[experiment]:
        base_pipeline["training"]["epochs_per_trigger"] = experiments[experiment]["epochs_per_trigger"]
    if "tail_triggers" in experiments[experiment]:
        base_pipeline["training"]["selection_strategy"]["config"]["tail_triggers"] = experiments[experiment]["tail_triggers"]
    print(experiment, base_pipeline)

    with open(f"./benchmark/pipeline_queue/{experiment}.yaml", "w") as f:
        yaml.dump(base_pipeline, f)

def send_message_telegram(message_text):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id=" + CHAT_ID_TELEGRAM + "&parse_mode=MarkdownV2&text="
    url = url + urllib.parse.quote_plus(message_text)
    if not requests.get(url).json()["ok"]:
        print("Error while sending the message",message_text,  requests.get(url).json())

def send_file_telegram(file_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    data = {
        'chat_id': CHAT_ID_TELEGRAM,
    }

    try:
        files = {
            'document': open(file_path, 'rb')
        }
    except FileNotFoundError:
        telegram_file = file_path.replace('.', '\\.')
        telegram_file = telegram_file.replace('_', '\\_')
        send_message_telegram(f"File {telegram_file} not found")
        return

    response = requests.post(url, data=data, files=files)
    if not response.json()["ok"]:
        print("Error while sending the file.")

import subprocess

def run_bash_script_experiments(experiment_name, experiment_name_telegram):
    script_path = "/local/home/fdeaglio4/modyn/run_experiment.sh"
    try:
        # Construct the command to run the Bash script with the custom parameter
        command = ["bash", script_path, experiment_name]
        send_message_telegram(f"*{experiment_name_telegram}*: Launching bash script")
        # Run the command and wait for completion
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        process.wait()


        # Check the return code to determine if the script was successful
        return_code = process.returncode
        if return_code == 0:
            send_message_telegram(f"*{experiment_name_telegram}*: Bash script completed successfully\\.")
            print("Standard Output:")
            print(stdout.decode("utf-8"))
        else:
            send_message_telegram(f"*{experiment_name_telegram}*: Bash script failed\\.")
            print("Standard Error:")
            print(stderr.decode("utf-8"))
            print("Standard Output:")
            print(stdout.decode("utf-8"))

    except Exception as e:
        print("An error occurred:", str(e))


def send_stats_message(experiment, training_time, experiment_telegram):
    try:
        with open(f"/scratch/fdeaglio/test_out_backup/{experiment}.pkl", "rb") as f:
            data = pickle.load(f)
        accuracies = [sum(data_) / len(data_) for data_ in data]
        avg_accuracy = str(sum(accuracies) / len(accuracies)).replace(".", "\\.")
        accuracies = str(accuracies).replace(".", "\\.")
        training_time = str(training_time).replace(".", "\\.")

        text = f"_{experiment_telegram}_\n\n*Average Accuracy*: {avg_accuracy}\n*Training time*: {training_time}\n*Accuracies*: {accuracies}"
        send_message_telegram(text)
    except Exception as e:
        print(e)
        send_message_telegram(f"An error occurred when processing the stats for {experiment_telegram}")


def main():
    with open(EXPERIMENTS_FILE, "r") as f:
        experiments = json.load(f)

    for experiment in experiments:
        experiment_telegram = experiment.replace("_", "\\_")
        create_yaml_experiment_file(experiment, experiments)
        send_message_telegram(f"*{experiment_telegram}*: Yaml file created")
        start_time = time.time()
        run_bash_script_experiments(experiment, experiment_telegram)
        end_time = time.time()
        elapsed_time = end_time - start_time
        send_stats_message(experiment, elapsed_time, experiment_telegram)
        send_file_telegram(f"/scratch/fdeaglio/test_out_backup/{experiment}.pkl")

if __name__ == "__main__":
    main()