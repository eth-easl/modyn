import requests
import sys
BOT_TOKEN = "6655968162:AAFNVdebd2iuJ5qHXL3GY2g3fMGcz2HCn9w"
CHAT_ID_TELEGRAM = "881642944"


def main():
    send_file_telegram(sys.argv[1])


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
        return

    response = requests.post(url, data=data, files=files)
    if not response.json()["ok"]:
        print("Error while sending the file.")

if __name__ == "__main__":
    main()