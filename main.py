from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os 
load_dotenv()

def generate_pet_name(animal_type="cat", pet_color="black"):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GEMINI_API_KEY"))

    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {pet_color} pet {animal_type} and want a cool name for it. Something that makes it sound like the ultimate ferocious and loyal guard dog. Any ideas?"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")

    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
   

    return response

if __name__ == "__main__":
    # list of tuples of animal pet types and color

    animals = [
        ("cat", "black"),
        ("dog", "brown"),
    ]
    for animal in animals:
        print(generate_pet_name(animal_type=animal[0], pet_color=animal[1]))
        print("")

    #print(generate_pet_name())