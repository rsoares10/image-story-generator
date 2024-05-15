from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())

# Image2Text
def img2text(url):
  image_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')
  text = image_to_text(url)[0]['generated_text']
  return text
 
if __name__ == '__main__':
  text = img2text('./data/img.jpg')
  print(f'\n-> {text}')