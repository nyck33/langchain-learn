'''
/home/nyck33/Documents/DataEngineering/Snowflake-The-Definitive-Guide-1.pdf:
'''
#%%
from langchain.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

#%%
load_dotenv()  # Ensure environment variables are loaded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#print every other char of gemini api key
print(GEMINI_API_KEY[::2])


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# %%
# Use HuggingFaceEmbeddings instead of GoogleGenerativeAIEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings

#embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

#%%
def create_vector_db_from_pdf(pdf_path: str) -> FAISS:
    try:
        loader = TextLoader(pdf_path)
        document = loader.load()
        print("PDF Loaded Successfully")  # Debugging statement
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(document)
    print(f"Number of documents after splitting: {len(docs)}")  # Debugging statement

    # Check the content of the first few documents
    for i, doc in enumerate(docs[:3]):
        print(f"Document {i}: {doc.page_content[:200]}...")  # Print first 200 characters of each document

    db = FAISS.from_documents(docs, embeddings)

    return db
# %%


#%%
def get_response_from_query(db, query, k):
    if db is None:
        print("Database is not created properly.")
        return None

    try:
        docs = db.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return None

    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful data engineering coach who will answer questions based on the uploaded PDF and your knowledge of the Snowflake.  
        
        Answer the following question: {question}
        By searching the following PDF document: {docs}
        
        Only use the factual information from the transcript to answer the question.
        Your answers should quote the source as much as possible and be in an academic tone
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
    except Exception as e:
        print(f"Error in generating response: {e}")
        return None

    return response, docs

#%%
text_path = "/home/nyck33/Documents/DataEngineering/Snowflake-The-Definitive-Guide.txt"

lines = []
with open(text_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()
#%%
print(lines[:10])
print(len(lines))
#%%

output_path = "/home/nyck33/Documents/DataEngineering/Snowflake-Lines.txt"
with open(output_path, 'w', encoding='utf-8-sig') as output_file:
    output_file.writelines(lines)

#%%
text_path = "/home/nyck33/Documents/DataEngineering/Snowflake-Lines.txt"

db = create_vector_db_from_pdf(text_path)

if db is not None:
    print("Vector database created successfully.")
else:
    print("Failed to create vector database.")

#%%
# write some code to pull some references out of the vector database

#%%

query = "What is Snowflake?"
response, docs = get_response_from_query(db, query, k=4)  # Adjust 'k' as needed based on performance and relevance

if response:
    print(response)
else:
    print("Failed to generate response.")


# %%
