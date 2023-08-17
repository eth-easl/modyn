import os
import time
from datetime import datetime

import requests
import urllib.parse
chat_id_tg = "881642944"

def _handle_not_sent(not_sent_counter):
    if not_sent_counter < 5:
        return
    url = "https://api.telegram.org/bot6655968162:AAFNVdebd2iuJ5qHXL3GY2g3fMGcz2HCn9w/sendMessage?chat_id=" + chat_id_tg + "&parse_mode=MarkdownV2&text="
    url = url + urllib.parse.quote_plus(f"No updates in the last {not_sent_counter} polls")
    requests.get(url)

    if not_sent_counter == 40:
        url = "https://api.telegram.org/bot6655968162:AAFNVdebd2iuJ5qHXL3GY2g3fMGcz2HCn9w/sendMessage?chat_id=" + chat_id_tg + "&parse_mode=MarkdownV2&text="
        url = url + urllib.parse.quote_plus(f"Shutting down since no updates in the last {not_sent_counter} polls")
        requests.get(url)
        exit()


def main():
    last_trigger_sent = -1
    not_sent_counter = 0
    last_time_sent = datetime.now()
    while True:
        time.sleep(30)
        elements = [int(el.split("_")[1]) for el in os.listdir("/scratch/fdeaglio/trigger_samples/")]
        if len(elements) == 0:
            not_sent_counter += 1
            _handle_not_sent(not_sent_counter)
            continue
        last_trigger = max(elements)

        if last_trigger == last_trigger_sent:
            not_sent_counter += 1
            _handle_not_sent(not_sent_counter)
            continue

        if last_trigger < last_trigger_sent:
            print("Reset!")

        delta = (datetime.now() - last_time_sent).total_seconds()
        time_str = str(round(delta, 2)).replace('.', '\\.')
        url = "https://api.telegram.org/bot6655968162:AAFNVdebd2iuJ5qHXL3GY2g3fMGcz2HCn9w/sendMessage?chat_id=" + chat_id_tg + "&parse_mode=MarkdownV2&text="
        url = url + urllib.parse.quote_plus(f"Training trigger {last_trigger}\\. Took {time_str}s")
        last_time_sent = datetime.now()

        requests.get(url)
        not_sent_counter = 0
        last_trigger_sent = last_trigger



if __name__ == "__main__":
    main()