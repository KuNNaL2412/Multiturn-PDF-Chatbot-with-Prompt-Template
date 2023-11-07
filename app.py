import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

with st.sidebar:
    st.title('🗨️ PDF Based Chatbot')
    st.markdown('''
    ## About App:

    The app's primary resource is utilised to create:

    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/kunal-pamu-710674230/)
    
    ''')
    st.write("Made by Kunal Shripati Pamu")
    
def main():
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.header("Multiturn PDF Chatbot")

        # upload the pdf
        pdf = st.file_uploader("Upload Your PDF:", type="pdf")

        # extract the text
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # split text into chunks
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            
            # check cache for pdf name and if present use the previous embeddings else create new ones 
            store_name = pdf.name[:-4]
            # st.write(f'{store_name}')
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
            else:
                embeddings=OpenAIEmbeddings()
                vector_store=FAISS.from_texts(chunks,embedding=embeddings)
                with open(f"{store_name}.pkl","wb") as f:
                    pickle.dump(vector_store,f)
            
            # Read user query
            query = st.text_input("Ask question about your PDF file:")
            
            # Initialize retriever, memory and llm for chat history retrieval, storing memory & using OpenAI llm
            retriever = vector_store.as_retriever(search_kwargs=dict(k=1))
            memory = VectorStoreRetrieverMemory(retriever=retriever)
            llm = OpenAI(temperature=0)
            
           # Create a default Template to set it's nature & boundary of the bot
            DEFAULT_TEMPLATE = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            
            Relevant pieces of previous conversation:
            {history}

            Conversation:
            Human: {input}
            Bot:"""
            PROMPT = PromptTemplate(
                input_variables=["history", "input"], template=DEFAULT_TEMPLATE
            )

            # Create a Conversation chain
            conversation_chain = ConversationChain(
                llm=llm,
                prompt=PROMPT,
                memory=memory,
                verbose=True
            )

            if query:
                # Display Result using the Conversation Chain
                response=conversation_chain.predict(input=query)
                st.write(response)

if __name__ == '__main__':
    main()