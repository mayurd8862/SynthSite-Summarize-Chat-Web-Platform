import streamlit as st
# from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
import os
import getpass
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] == st.secrets["GOOGLE_API_KEY"]

header ={
    'GOOGLE_API_KEY' : st.secrets["GOOGLE_API_KEY"]
}

llm = ChatGoogleGenerativeAI(model="gemini-pro")


map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
combine_prompt = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
also add emojis whereever need.
```{text}```
BULLET POINT SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])


def main():
    # Page title
    st.title("🚀Summarize Website Content")
    
    # Input: YouTube URL
    url = st.text_input("Enter YouTube Video URL:")

    if st.button("Summarize"):

        loader = WebBaseLoader(url)
        web_data = loader.load()

        # summarization chain with extra prompt
        summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
#                                      verbose=True
                                    )
        
        output = summary_chain.invoke(web_data)

        # Display summary
        st.subheader("Website Summary:")
        st.write(output['output_text'])

if __name__ == "__main__":
    # test()
    main()
