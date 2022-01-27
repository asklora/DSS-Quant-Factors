import yagmail
send_from = "asklora@loratechai.com"
send_pwd = "lzlztzvrndfinjdy"

def send_mail(subject, text, file=None, send_to="clair.cui@loratechai.com"):
    yag = yagmail.SMTP(send_from, send_pwd)
    contents = [text, file]
    yag.send(send_to, subject, contents)

if __name__=="__main__":
    send_to = "clair.cui@loratechai.com"
    subject = "test"
    text = "test"
    file = "missing_by_ticker_USD.xlsx"
    send_mail(subject, text, file, send_to)