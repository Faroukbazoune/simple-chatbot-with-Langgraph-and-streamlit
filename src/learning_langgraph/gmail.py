import os
import base64
from langchain.messages import AIMessage
from langgraph.types import Command, interrupt
from email.message import EmailMessage
from google.auth.transport.requests import Request  # refreshing the token
from google.oauth2.credentials import Credentials  # for creds
from google_auth_oauthlib.flow import (
    InstalledAppFlow,
)  # so the authentification window pop up
from googleapiclient.discovery import build  # to build the service

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def getting_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

            with open("token.json", "w") as token:
                token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(to: str, body: str, subject: str):
    service = getting_gmail_service()

    message = EmailMessage()
    message.set_content(body)
    message["to"] = to
    message["subject"] = subject

    decision = interrupt(f"do yo want to send this message to {to} \n\n{body}")
    if not decision["approved"]:
        return "not approved"
    else:

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        result = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": encoded_message})
            .execute()
        )

        return result["id"]
