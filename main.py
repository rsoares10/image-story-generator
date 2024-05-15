import os

import requests
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, OpenAI
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Image to text
def img2text(url):
  image_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')
  text = image_to_text(url)[0]['generated_text']
  return text

#LLM step
def generate_story(scenario):
  template = '''
  You are a story teller:
  You can generate a short story based on a simple narrative, the story should be no more than 20 words.

  CONTEXT: {scenario}
  STORY:
  '''

  prompt = PromptTemplate(template=template, input_variables=['scenario'])
  story_llm = LLMChain(llm=OpenAI(model_name='gpt-3.5-turbo', temperature=1), prompt=prompt, verbose=True)
  story = story_llm.predict(scenario=scenario)
  return story

# Text to epeech
def text2speech(message):
  API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
  headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
  paylods = {'inputs': message}
  response = requests.post(API_URL, headers=headers, json=paylods)
  with open('./output/audio.flac', 'wb') as file:
    file.write(response.content)
 
if __name__ == '__main__':
  description = img2text('./data/img.jpg')
  story = generate_story(description)
  text2speech(story)