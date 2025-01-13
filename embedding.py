
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

gemini_key=os.getenv("gemini_api_key")

def create_embeddings(file_content,file_name):
    embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path="D:\Python_Codes\RAG_App") 
    collection=client.get_or_create_collection(name=f"collection_{file_name}",metadata={"hnsw:space":"cosine"})

    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! "],  # List of characters to split on
    chunk_size=2000,  # The maximum size of your chunks
    chunk_overlap=400,  # The maximum overlap between chunks
    )

    text={}
    text['chunk']=text_splitter.create_documents([file_content])
    for chunk_id,chunk in enumerate(text["chunk"]):
        collection.add(
            documents=[chunk.page_content],
            ids=[f"chunk_id"],
            metadatas={"title":file_name}
            )
        
    return collection




def get_final_answer(user_query,file_content,collection,n_results,file_name):

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    search_results = collection.query(query_texts=[user_query], n_results=n_results)
    result_str = ""
    response=""
    for result in search_results["documents"][0]:
    
        final_prompt= f"""
                        Instructions:
                        Answer the user question mentioned below based on search results provided

                        At the end of your answer, cite the URL of the search result your answer draws from. Use the following format:
                        <Your answer here>. Source: <URL of the search result your answer comes from here>. Output only the answer part and reference URL

                        User question: <{user_query}>

                        Search Results: <{result}>

                        Source:<{file_name}>
                        Your answer:
                        """
    
        
        response += model.generate_content(final_prompt).text
    return response
