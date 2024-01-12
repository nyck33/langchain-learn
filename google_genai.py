#%%
import google.generativeai as genai 
from dotenv import load_dotenv
import os

#%%
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#%%
genai.configure(api_key=GEMINI_API_KEY)

#%%
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
        

# %%
model = genai.GenerativeModel('gemini-pro')

#%%
prompt = """
What are the best ways to learn Korean if I am living in Tokyo and on a low budget?
"""

#%%
response = model.generate_content(prompt)

#%%
from IPython.display import Markdown
from IPython.core.display import display

import pathlib, textwrap

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
# %%
to_markdown(response.text)
# %%
