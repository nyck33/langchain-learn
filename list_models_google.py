#%%
import pathlib
import textwrap

import google.generativeai as genai

# Used to securely store your API key
from google.colab import userdata

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

#%%
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

#%%
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

#%%

