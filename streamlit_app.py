import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
import speech_recognition as sr 

# Streamlit title
st.title("Medical AI Chatbot Assistant")
st.image('chatbot.jpg')

# Define language options
language_options = {"English": "en", "Telugu": "te", "Tamil": "ta"}
selected_language = st.selectbox("Select your preferred language for answers:", list(language_options.keys()))

# Translator initialization for multilingual support
translator = GoogleTranslator()

# Define folder path for vector store
folder_path = "db"

# Initialize models and embeddings
cached_llm = Ollama(model="wizardlm2:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

# Define the prompt with a medical context disclaimer
raw_prompt = PromptTemplate.from_template(
    """<s>[INST] You are a medical assistant. Answer with reliable medical advice if known, otherwise respond 'I'm not sure about this'. [/INST] </s>
    [INST] {input} Context: {context} Answer: [/INST]"""
)

# Function to retrieve answer and handle language translation
def retrieve_answer(query):
    st.write("Loading vector store...")
    try:
        # Load vector store for retrieval
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None

    st.write("Creating chain...")
    try:
        # Create retriever and document processing chain
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 20, "score_threshold": 0.6}
        )
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        # Retrieve response with user's query
        result = chain.invoke({"input": query})
        answer = result["answer"]

        # Translate answer if the selected language is not English
        if selected_language != "English":
            answer_translated = translator.translate(answer, dest=language_options[selected_language])
            answer = answer_translated  # Override with translated answer

        # Extract sources for display
        sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
        return answer, sources
    except Exception as e:
        st.error(f"Error creating chain: {e}")
        return None, None

# Upload PDF document section in Streamlit app
st.header("Upload Medical Document")

# Allow user to upload a PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    save_file = f"pdf/{uploaded_file.name}"
    with open(save_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

    # Load PDF document for processing
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    chunks = text_splitters.split_documents(docs)
    
    st.write("Embedding medical document...")
    try:
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=folder_path
        )
        vector_store.persist()
        st.success("Document processed and embedded for retrieval.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Query section for chatbot interaction
st.header("Ask a Medical Question")

# Query input selection: Text or Voice input
query_input_method = st.selectbox("Select the method of input:", ["Text Input", "Voice Input"])

query = ""

if query_input_method == "Text Input":
    query = st.text_input("Enter your medical question here")
    
elif query_input_method == "Voice Input":
    st.write("Click on the button and speak your query")
    recognizer = sr.Recognizer()

    if st.button("Start Recording"):
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = recognizer.listen(source)  # Capture audio

            try:
                # Recognize the speech and convert it to text
                query = recognizer.recognize_google(audio, language=language_options[selected_language])
                st.write("Recognized text:", query)

                # Translate non-English input to English
                if selected_language != "English":
                    query = translator.translate(query, dest="en").text

            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
            except sr.RequestError:
                st.error("There was an issue with the speech recognition service.")

if st.button("Submit"):
    if query:
        # Retrieve answer and sources
        answer, sources = retrieve_answer(query)
        
        # Display answer in selected language
        if answer:
            st.write(f"**Answer in {selected_language}:**", answer)

            # Display source documents
            st.write("**Sources:**")
            for source in sources:
                st.write(f"- Source: {source['source']}, Content: {source['page_content']}")
        else:
            st.warning("No answer found for this query. Please try rephrasing.")
    else:
        st.warning("Please enter a query.")
