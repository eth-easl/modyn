# isort: skip_file

import os
import random
import time
from . import istarmap  # fmt: skip  # noqa  # isort:skip

from multiprocessing import Manager, Pool

import requests
import tqdm


# This scraper is based on https://github.com/drimpossible/ACM/blob/main/scripts/cglm_scrape.py

TOTAL_MACHINES = 1
MACHINE_ID = 0

OUTPUT_DIR = "./metadata/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PATH_TO_TRAINATTRIBUTIONS = "./train_attribution.csv"

ids = []

fr = open(PATH_TO_TRAINATTRIBUTIONS)
lines = fr.readlines()
fr.close()

# Get photo_id, line_match fields from the file
for num_line, line in enumerate(lines):
    if num_line % TOTAL_MACHINES == MACHINE_ID:
        photo_id = line.strip().split(",")[1].split(":")[-1].strip()
        line_match = line.strip().split(",")[0].strip()
        ids.append((photo_id, line_match))


# Function to save failed items to disk
def save_failed_items(failed_items):
    with open(os.path.join(OUTPUT_DIR, "failed_items.txt"), "w") as f:
        for item in failed_items:
            f.write(f"{item[0]},{item[1]}\n")
    # print(f"Failed items persisted to '{os.path.join(OUTPUT_DIR, 'failed_items.txt')}'.")


# Function to download and store the metadata
def download_metadata(id, failed_items):
    photo_id, line_match = id
    delay = 3  # Initial delay in seconds
    max_retries = 3
    timeout = 10  # Timeout in seconds
    output_path = os.path.join(OUTPUT_DIR, line_match + ".txt")
    if os.path.isfile(output_path):
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    for retry in range(max_retries):
        time.sleep(random.uniform(0.2, 2.5))
        try:
            # You might want to host the server yourself.
            # res = requests.get('http://localhost:9000/public_html/commonsapi.php?meta&image='+photo_id, timeout=timeout, headers=headers)
            res = requests.get(
                "https://opendata.utou.ch/glam/magnus-toolserver/commonsapi.php?meta&image=" + photo_id,
                timeout=timeout,
                headers=headers,
            )

            if res.status_code == 200:
                content = res.content.decode("utf8")
                if "HTTP/1.1 429" in content:
                    print("Rate limited in response content. Retrying...")
                    time.sleep(delay)
                    delay *= 2  # Double the delay for rate limiting in response content
                    continue

                with open(output_path, "w") as f:
                    f.write(content + "\n")
                    f.flush()

                time.sleep(random.uniform(0.2, 1.5))
                return
            elif res.status_code == 429:
                retry_after = int(res.headers.get("Retry-After", str(delay)))
                delay = max(delay, retry_after * 2)  # Double the delay suggested by the server
                print(f"Rate limited. Retrying in {delay} seconds.")
                time.sleep(delay)
            else:
                print(f"Request failed with status code {res.status_code}. Retrying...")
                time.sleep(delay)
                delay *= 2  # Double the delay for other failed status codes
        except requests.exceptions.Timeout:
            # print(f"Request timed out. Retrying...")
            time.sleep(delay)
            delay *= 2  # Double the delay for timeouts
        except Exception as e:
            print(f"Error occurred: {str(e)}. Retrying...")
            time.sleep(delay)
            delay *= 2  # Double the delay for exceptions

    failed_items.append(id)
    save_failed_items(failed_items)  # Save failed items to disk


if __name__ == "__main__":
    manager = Manager()
    failed_items = manager.list()
    progress = manager.Value("i", 0)

    pool_size = 16  # Adjust this number if necessary
    with Pool(pool_size) as pool:
        print(f"Pool with {pool_size} workers initialized.")

        inputs = [(id, failed_items) for id in ids]
        for _ in tqdm.tqdm(pool.istarmap(download_metadata, inputs), total=len(inputs)):
            pass

    # Persist the list of failed items to disk at the end
    save_failed_items(failed_items)
