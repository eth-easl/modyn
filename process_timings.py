import datetime
import os.path
import sys
import json

import dateutil.tz as tz
zh = tz.gettz('Europe/Zurich')

def get_times(event, duration):
    name = None
    if event == "Event.PREPROCESS_END":
        name = "Preprocessing"
    elif event == "Event.FORWARD_END":
        name = "Forward Pass"
    elif event == "Event.BACKWARD_END":
        name = "Backward Pass"
    elif event == "Event.STEP_END":
        name = "Step"
    elif event == "Event.START_BATCH":
        name = "Data Loading"

    if name is None:
        return None
    return {"Event": name, "Duration": duration.total_seconds()}

def group_by(name, times, reduction = sum, clean = False):
    return round(reduction(el["Duration"] for el in times if el["Event"] == name and (el["Duration"] < 0.75 if clean else True)), 5)

def process_file(file):
    f = open(file, "r")
    times = []
    for line in f:
        splitted_line = line.split()
        timestamp = datetime.datetime.fromisoformat(splitted_line[0]).replace(tzinfo=zh)
        event = splitted_line[1]
        if event != "Event.START_LOGGING":
            logged_event = get_times(event, timestamp - previous_timestamp)
            if logged_event:
                times.append(logged_event)
        previous_timestamp = timestamp
    return {
        "Total forward" : group_by("Forward Pass", times),
        "Total forward cleaned" : group_by("Forward Pass", times, clean = True),
        "Total backward" : group_by("Backward Pass", times),
        "Total backward cleaned" : group_by("Backward Pass", times, clean=True),
        "Total step" : group_by("Step", times),
        "Total step cleaned" : group_by("Step", times, clean=True),
        "Total data loading" : group_by("Data Loading", times),
        "Total data loading cleaned" : group_by("Data Loading", times, clean=True),
        "Total preprocessing" : group_by("Preprocessing", times),
        "Total preprocessing cleaned" : group_by("Preprocessing", times, clean=True),
    }

def get_sum_total(data):
    ret = {}
    for key in data["timing_trigger_0.txt"].keys():
        ret[key] = sum(data[file][key] for file in data)

    return ret



def main():
    all_files = {}


    for file in os.listdir(sys.argv[1]):
        if file.startswith("timing"):
            all_files[file] = process_file(os.path.join(sys.argv[1], file))

    all_files["total"] = get_sum_total(all_files)

    with open(os.path.join(sys.argv[1], "processed.json"), "w") as f:
        json.dump(all_files, f)


if __name__ == "__main__":
    main()