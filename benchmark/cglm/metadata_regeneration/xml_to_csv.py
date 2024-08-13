import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from xml.etree import ElementTree

import pandas as pd


def process_file(file_path):
    try:
        file_id = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path) as file:
            xml_content = file.read()

        root = ElementTree.fromstring(xml_content)

        if root.tag == "error":
            error_message = root.text.strip()
            if error_message.startswith("File does not exist"):
                file_not_found.append(file_id)
            else:
                parsing_errors.append(file_id)

            return None

        file_element = root.find("file")
        if file_element is None:
            parsing_errors.append(file_id)
            return None

        name = file_element.find("name").text
        upload_date_element = file_element.find("upload_date")

        if upload_date_element is None:
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            no_upload_date.append(file_id)
            return None

        upload_date = datetime.strptime(upload_date_element.text, "%Y-%m-%dT%H:%M:%SZ").timestamp()

        return file_id, name, upload_date

    except ElementTree.ParseError:
        # Try to extract upload_date using regular expression
        match = re.search(
            r"<upload_date>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)</upload_date>",
            xml_content,
        )
        if match:
            upload_date = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ").timestamp()
            return file_id, None, upload_date
        else:
            parsing_errors.append(file_id)
            return None

    except ValueError as e:
        # Handle ValueError when parsing upload_date
        if "time data" in str(e):
            no_upload_date.append(file_id)
        else:
            parsing_errors.append(file_id)
        return None

    except Exception:
        # Handle other exceptions
        parsing_errors.append(file_id)
        return None


def process_directory(directory):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".txt")]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, file_paths))

    results = [result for result in results if result is not None]

    if results:
        fileids, filenames, upload_dates = zip(*results)
        df = pd.DataFrame({"fileid": fileids, "filename": filenames, "upload_date": upload_dates})
        df.to_csv("output.csv", index=False)
    else:
        print("No valid data found.")

    with open("file_not_found.txt", "w") as file:
        file.write("\n".join(file_not_found))

    with open("no_upload_date.txt", "w") as file:
        file.write("\n".join(no_upload_date))

    with open("parsing_errors.txt", "w") as file:
        file.write("\n".join(parsing_errors))


if __name__ == "__main__":
    directory = "/scratch/maximilian.boether/scrape/metadata"
    file_not_found = []
    no_upload_date = []
    parsing_errors = []

    process_directory(directory)
