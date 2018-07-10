from flask_mail import Message, Mail

# from project import app, mail

mail = Mail()


def send_email(to, subject, template):
    msg = Message(
        subject,
        recipients=[to],
        html=template,
        sender="registration@malgazer.com"
    )
    mail.send(msg)
