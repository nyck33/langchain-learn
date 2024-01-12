from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS 
from dotenv import load_dotenv

import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#video_url = "https://youtu.be/lG7Uxts9SXs?si=y9TMw0Fw3PZM_D77"

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings) 

    return db


def get_response_from_query(db, query, k):
    # up to 32k tokens, approx 8000 words
    #sending chunk size of 1k, so can send up to 32 chunks, k = 32?, demo video uses 4 for max 4097 tokens
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response, docs


if __name__=="__main__":
    video_url="https://youtu.be/O2lZkr-aAqk?si=PhRSg6uYkYUBQJ01"
    db = create_vector_db_from_youtube_url(video_url)
    query = "Summarize what category theory is?"
    response, docs = get_response_from_query(db, query, k=4)
    print(response)
    

