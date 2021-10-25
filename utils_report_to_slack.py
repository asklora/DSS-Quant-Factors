from slack_sdk import WebClient
import logging

SLACK_API = "xoxb-305855338628-1139022048576-2KsNu5mJCbgRGh8z8S8NOdGI"
default_channel = "#factor_message"
logger = logging.getLogger(__name__)

def report_to_slack(message, channel=default_channel):

    try:
        client = WebClient(token=SLACK_API, timeout=30)
        client.chat_postMessage(
            channel=channel,
            text=message,
        )

    except Exception as e:
        print(e)

def report_series_to_slack(message=None, df=None, id=None):

    if message:
        report_to_slack(message)

    message = "```"
    for k, v in df.to_dict().items():
        message += f"{k.ljust(40)}{v}\n"
    message += "```"
    print(message)

    if id:
        report_to_slack_user(message, id=id)
    else:
        report_to_slack(message)

def report_df_to_slack(message, df, id=None):
    if message:
        report_to_slack(message)

    message = "```"
    message += f"{'columns'.ljust(20)}"

    for i in df.columns.to_list():  # add columns
        message += f"{i.ljust(10)}"
    message += "\n"

    for k, v in df.transpose().to_dict(orient='list').items():
        message += f"{str(k).ljust(20)}"
        for i in v:
            message += f"{str(i).ljust(10)}"
        message += "\n"

    message += "```"
    print(message)

    if id:
        report_to_slack_user(message, id=id)
    else:
        report_to_slack(message)

def file_to_slack(file, filetype, title, channel=default_channel):

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

def file_to_slack_user(file, filetype, title, id='U026B04RB3J'):

    try:
        client = WebClient(token=SLACK_API, timeout=30)
        result = client.files_upload(
            channels=id,
            file=file,
            filetype=filetype,
            title=title)
        logger.info(result)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    # file_to_slack_user('test')
    report_to_slack('test', channel='U026B04RB3J')