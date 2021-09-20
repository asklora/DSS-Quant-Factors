from slack_sdk import WebClient
import logging

SLACK_API = "xoxb-305855338628-1139022048576-2KsNu5mJCbgRGh8z8S8NOdGI"
channel = "#factor_message"
logger = logging.getLogger(__name__)

def report_to_slack(message):

    try:
        client = WebClient(token=SLACK_API, timeout=30)
        api_response = client.api_test()
        client.chat_postMessage(
            channel=channel,
            text=message)

    except Exception as e:
        print(e)

def file_to_slack(file, filetype, title):

    try:
        client = WebClient(token=SLACK_API, timeout=30)
        result = client.files_upload(
            channels=channel,
            file=file,
            filetype=filetype,
            title=title)
        logger.info(result)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    report_to_slack('test')