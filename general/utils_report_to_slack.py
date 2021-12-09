from slack_sdk import WebClient

class to_slack:
    def __init__(self, channel="#factor_message"):
        self.SLACK_API = "xoxb-305855338628-1139022048576-2KsNu5mJCbgRGh8z8S8NOdGI"
        self.slack_name_to_id = {
            "#factor_message": "#factor_message",
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

    def message_to_slack(self, message):

        try:
            client = WebClient(token=self.SLACK_API, timeout=30)
            client.chat_postMessage(
                channel=self.channel,
                text=message,
            )

        except Exception as e:
            print(e)

    def series_to_slack(self, message=None, df=None):

        if message:
            self.message_to_slack(message)

        message = "```"
        for k, v in df.to_dict().items():
            message += f"{k.ljust(40)}{v}\n"
        message += "```"
        print(message)
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
        print(message)
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
            print(e)

if __name__ == "__main__":
    # file_to_slack_user('test')
    to_slack("clair").message_to_slack('test')