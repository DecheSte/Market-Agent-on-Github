import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import markdown
import textwrap

load_dotenv(override=True)

def send_market_report_email(receiver_email: str, subject: str, markdown_content: str):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        return False

    try:
        # HTML
        cleaned_markdown = textwrap.dedent(markdown_content).strip()
        html_body = markdown.markdown(cleaned_markdown)

        # CSS
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
                h2 {{ color: #34495e; margin-top: 20px; }}
                ul {{ padding-left: 20px; }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """

        # Construct E-mail
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = receiver_email

        html_part = MIMEText(html_content, "html", "utf-8")
        message.attach(html_part)

        # Deliver
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())

        print(f"Successfully sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"Fail to send: {e}")
        return False