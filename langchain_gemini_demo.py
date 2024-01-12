# from https://www.analyticsvidhya.com/blog/2023/12/google-gemini-api/

#%%
#from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
#%% 
from dotenv import load_dotenv
import os
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

#%%
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key) 

response = llm.invoke(
        "What are the best ways to learn Korean if I am living in Tokyo and on a low budget?"
    )


#%%
from IPython.display import Markdown
from IPython.core.display import display

import pathlib, textwrap

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

#%%
to_markdown(response.content)



# %%
