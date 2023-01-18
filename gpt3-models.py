import os
import openai
openai.organization = "org-ux46YI3bUix2XtpSd8761gtu"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()