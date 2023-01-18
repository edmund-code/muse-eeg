import os
import openai

# Import the required module for text to speech conversion
import pyttsx3

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org = os.getenv("OPENAI_ORG")
openai.organization = openai_org
openai.api_key = openai_api_key

context = "Patient: Good morning Doctor.\nDoctor: Good morning! You seem pale and your voice sounds different."
x = 1

while x == 1:
  sentiment = input("sentiment: ")
  keyword = input("keyword: ")

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=" (" + context + ") \n Q: Write a response to the Doctor as a Patient using the given words in a (" + sentiment + ") tone : (" + keyword + "):",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  gpt_response_text = response.choices[0].text

  
  # init function to get an engine instance for the speech synthesis
  engine = pyttsx3.init()

  newVoiceRate = 80
  engine.setProperty('rate',newVoiceRate)
  
  # say method on the engine that passing input text to be spoken
  #engine.say(gpt_response_text)
  print(gpt_response_text)
  # run and wait method, it processes the voice commands.
  engine.runAndWait()


  response = input("Response: ")

  if response == "EXIT":
      x = x + 1

  context = "%s Patient:%s Doctor:%s" % (context, gpt_response_text, response)
  print(context)