import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader#pip install  -U langchain-community
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64#file encoding

#moel and tokenizer
# Load model from hugging face
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
base_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

#preprocessing the pdf file
def file_pre(file):
    loader= PyPDFLoader(file)
    pages=loader.load_and_split()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    texts=text_splitter.split_documents(pages)
    final_texts=""
    for text in texts:
        print(text)
        final_texts=final_texts + text.page_content
    return final_texts

#llm pipeline
def llm_pipline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50

    )
    input_text=file_pre(filepath)
    result=pipe_sum(input_text)
    result=result[0]['summary_text']
    return result

#func to display pdf 

@st.cache_data
def displaypdf(file):
    #open pdf file
    with open(file,"rb") as f:
        base64_pdf=base64.b64encode(f.read()).decode('utf-8')
    #embedding pdf in html
    pdf_display=F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    #diplaying pdf
    st.markdown(pdf_display,unsafe_allow_html=True)

st.set_page_config(layout='wide',page_title='Research Paper Summarizer')

def main():
    st.title('RP summarization using T5🤖')

    uploaded_file=st.file_uploader("Upload the research paper in pdf format",type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1,col2=st.columns(2)
            filepath=uploaded_file.name
            with open(filepath,'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("uploaded file: ")
                pdf_viewer=displaypdf(filepath)
            with col2:
                st.info("Summarization: ")
                summary=llm_pipline(filepath)
                st.success(summary)



        
if __name__=='__main__':
        main()






