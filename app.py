import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

# Load environment variables
load_dotenv()

# Creating session variables
if "unique_id'" not in st.session_state:
    st.session_state["unique_id"] = ""

def main():
    st.set_page_config(page_title="Resume Screening Assistance", page_icon="📝")
    st.title("Resume Screening Assistance")

    # Text area for job description
    job_description = st.text_area("Enter the 'JOB DESCRIPTION' here", key="1")

    # Text input for number of resumes to return
    document_count = st.text_input("Number of 'RESUMES' to return", key="2", placeholder="2")

    # Upload resumes
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

	# Enable the button only if all inputs are filled
    submit = False
    if job_description and document_count and pdf:
        submit = st.button("ANALYZE")
    else:
        st.warning("Please fill in all the inputs to enable analysis.")

    if submit:
        with st.spinner("Wait for it..."):
            try:
                # Create a unique ID for this session to filter out docs
                st.session_state["unique_id"] = str(uuid.uuid4().hex)

                # Create a list of documents from uploaded PDF files
                docs = create_docs(pdf, st.session_state["unique_id"])
                st.write("*Resumes uploaded* :" + str(len(docs)))

                # Create embeddings instance
                embeddings = create_embeddings_load_data()

                # Push data to Pinecone
                # docsearch = push_to_pinecone(docs, embeddings)
                docsearch = push_to_chromadb(docs, embeddings)
                relevant_docs = similarity_search(docsearch, job_description,document_count, st.session_state["unique_id"])

                # st.write(relevant_docs)
                st.write(":heavy_minus_sign:" * 30)

                # Display relevant documents
                for index, doc_info in enumerate(relevant_docs, start=1):
                    st.subheader(f"👉 Resume {index}")
                    st.write("**File** : " + doc_info[0].metadata['name'])

					#Expander to show more details
                    with st.expander('Click to open 👀'):
                        st.info("**Match Score** : " + str(doc_info[1]))
                        # Get summary using LLM
                        summary = get_summary(doc_info[0])
                        st.write("**Summary** : " + summary)
                st.success("I hope you find the right candidate.❤️")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# Invoking main function
if __name__ == '__main__':
    main()
