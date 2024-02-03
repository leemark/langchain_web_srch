# Import necessary libraries and modules for the application.
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables, useful for API keys and configuration.
load_dotenv()

def get_vectorstore_from_url(url):
    # Load document from URL using WebBaseLoader.
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the loaded document into chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vector store from the document chunks using Chroma and OpenAI embeddings.
    vectorstore = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vectorstore

def get_context_retriever_chain(vector_store):
    # Initialize ChatOpenAI for language model operations.
    llm = ChatOpenAI() 
    # Convert the vector store to a retriever for fetching relevant documents.
    retriever = vector_store.as_retriever()
    # Define a prompt template for generating search queries based on chat history.
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"), 
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    ])

    # Create a chain that integrates the retriever with the language model, guided by the prompt.
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm=ChatOpenAI()
    prompt= ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the following context:\n\n{context}\n\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    # Combine the retriever chain with the stuff documents chain for a comprehensive response generation.
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    # Fetch the retriever chain based on the session's vector store.
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
     # Fetch the conversational RAG chain using the retriever chain.
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Invoke the conversation chain with user input and chat history to get a response.
    response = conversation_rag_chain.invoke({
        "input": user_query,
        "chat_history": st.session_state.chat_history,
    })
    st.write(response)
    return response['answer']

# Configure the Streamlit page with title and icon.
st.set_page_config(page_title="Chat w/web", page_icon=":robot:")
st.title("Chat w/web")

# Setup sidebar for user settings, specifically for entering a website URL.
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("website URL")

# Check for website URL input and initialize session states as needed.
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, how can I help you today?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url) 

    # Capture user input from chat.
    user_query = st.chat_input("Type a message and press enter to send")

    # If user input is provided, generate and display the response.
    if user_query is not None and user_query != "":
        response = get_response(user_query)

        # Update session state to include the latest messages.
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display the chat history in the UI.
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
    
# built based on https://www.youtube.com/watch?v=bupx08ZgSFg 