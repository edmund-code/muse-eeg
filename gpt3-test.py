import os
import openai

# Import the required module for text to speech conversion
import pyttsx3

openai_api_key = "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" 

openai.api_key = openai_api_key

sentiment = "negative"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Q: how was your vacation?\nA (" + sentiment + "):",
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
engine.say(gpt_response_text)
 
# run and wait method, it processes the voice commands.
engine.runAndWait()