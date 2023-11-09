import os

import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

openai_api_key = os.getenv('OPENAI_KEY')

def generate_response(txt):
    # Instantiate the LLM model

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=700, chunk_overlap=100)
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # # Text summarization
    # chain = load_summarize_chain(llm, chain_type='map_reduce')
    # return chain.run(docs)
    map_prompt = """
    Escribe un resumen conciso de este texto:
    "{text}"
    Resumen conciso:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    combine_prompt = """
    Escribe un resumen conciso de este texto separado por triple commilas.
    Retorna tu respuesta en bullet points que cubran los puntos claves del texto.
    ```{text}```
    BULLET POINT RESUMEN:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                    )
    return summary_chain.run(docs)


def process_pdf(pdf_reader, pages):
    text = ""
    for page in pdf_reader.pages[pages[0]:pages[1]]:
        text += page.extract_text() + "\n"
    return text

# Page title
st.set_page_config(page_title='Que vagancia leer..')
st.title('🔗 Pasame un pdf que te lo resumo.')

# Text input
pdf_file = st.file_uploader("Tirar pdf aca", type=["pdf"])
st.info('Evitar subir manuscritos escaneados hace mas de mil años, la magia tiene su limite', icon="🙌")
if pdf_file is not None:
    pdf_reader = PdfReader(pdf_file)
    n_pages = len(pdf_reader.pages)

    # Create a slider for page selection
    pages = st.slider('paginas a resumir', 0, n_pages, (0, n_pages), step=1)
    st.write('Desde-hasta:', pages)


# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Resumir!', disabled=(not pdf_file))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Leyendo...'):
            text = process_pdf(pdf_reader, pages)
            st.title('Texto a resumir')
            if len(text) > 1000:
                st.warning('Mucho texto... toma los primeros 10000 caracteres')
                text = text[:10000]
            elif len(text) < 100:
                st.warning('Poco texto, no se que hacer con esto...')
            elif len(text) <= n_pages:
                st.warning('Totalmente ilegible, no se que hacer con esto...')
            st.markdown(text)
        with st.spinner('Resumiendo...'):
            response = generate_response(text)
            result.append(response)
            del openai_api_key

if len(result):
    st.title('Resumen')
    st.info(response)
