from slack_sdk import WebClient
from global_vars import logger, LOGGER_LEVEL


logger = logger(__name__, LOGGER_LEVEL)

class to_slack:
    def __init__(self, channel="#dss-quant-factors-message"):
        self.SLACK_API = "xoxb-305855338628-1139022048576-2KsNu5mJCbgRGh8z8S8NOdGI"
        self.slack_name_to_id = {
            "#dss-quant-factors-message": "#dss-quant-factors-message",
            "clair": "U026B04RB3J",
            "stephen": "U8ZV41XS9",
            "nick": "U01JKNY3D0U"
        }
        try:
            self.channel = self.slack_name_to_id[channel]
        except Exception as e:
            # report error to Clair
            self.channel = self.slack_name_to_id["clair"]
            self.message_to_slack(f"Send to slack ERROR: need to add slack user id for {channel} ("
                                 f"currently user_id includes: {list(self.slack_name_to_id.keys())})")
            raise Exception(e)

    def message_to_slack(self, message, trim_msg=True):
        if trim_msg:
            message = str(message)
        logger.info(message)
        try:
            client = WebClient(token=self.SLACK_API, timeout=30)
            client.chat_postMessage(
                channel=self.channel,
                text=message,
            )

        except Exception as e:
            logger.warning(e)

    def series_to_slack(self, message=None, df=None):

        if message:
            self.message_to_slack(message)

        message = "```"
        for k, v in df.to_dict().items():
            message += f"{k.ljust(40)}{v}\n"
        message += "```"
        logger.info(message)
        self.message_to_slack(message)

    def df_to_slack(self, message, df):
        if message:
            self.message_to_slack(message)

        message = "```"
        message += f"{'columns'.ljust(20)}"

        num_col = df.select_dtypes(float).columns.to_list()
        df[num_col] = df[num_col].round(2)
        for i in df.columns.to_list():  # add columns
            message += f"{i.ljust(10)}"
        message += "\n"

        for k, v in df.transpose().to_dict(orient='list').items():
            message += f"{str(k).ljust(20)}"
            for i in v:
                message += f"{str(i).ljust(10)}"
            message += "\n"

        message += "```"
        logger.info(message)
        self.message_to_slack(message)

    def file_to_slack(self, file, filetype, title):

        try:
            client = WebClient(token=self.SLACK_API, timeout=30)
            result = client.files_upload(
                channels=self.channel,
                file=file,
                filetype=filetype,
                title=title)

        except Exception as e:
            logger.warning(e)

if __name__ == "__main__":
    # file_to_slack_user('test')
    to_slack("clair").message_to_slack('test')