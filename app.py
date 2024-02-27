import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

# Load environment variables
load_dotenv()

# Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    st.set_page_config(page_title="Resume Screening Assistance", page_icon="📝")
    st.title("HR - Resume Screening Assistance 💁")
    st.subheader("I can help you in the resume screening process")

    # Text area for job description
    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here", key="1")

    # Text input for number of resumes to return
    document_count = st.text_input("No. of 'RESUMES' to return", key="2", placeholder="2")

    # Upload resumes
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

	# Enable the button only if all inputs are filled
    submit = False
    if job_description and document_count and pdf:
        submit = st.button("ANALYZE")
    else:
        st.warning("Please fill in all the inputs to enable analysis.")

    if submit:
        with st.spinner('Wait for it...'):
            try:
                # Create a unique ID for this session to filtrer out the uploaded
                st.session_state['unique_id'] = str(uuid.uuid4().hex)

                # Create a list of documents from uploaded PDF files
                docs = create_docs(pdf, st.session_state['unique_id'])
                st.write(st.session_state['unique_id'])
                st.write("*Resumes uploaded* :" + str(len(docs)))

                # Create embeddings instance
                embeddings = create_embeddings_load_data()

                # Push data to Pinecone
                docsearch = push_to_pinecone(docs, embeddings)
                relevant_docs = mmr_search(docsearch, job_description, st.session_state['unique_id'])

                # Fetch relevant documents from Pinecone
                # relevant_docs = similar_docs(job_description, document_count, embeddings, st.session_state['unique_id'])
                # relevant_docs = similarity(job_description, docsearch)
                # relevant_docs = similarity(job_description,docsearch,document_count, st.session_state['unique_id'])
                # st.write(docsearch)
                st.write(relevant_docs)

				# line seperator
                st.write(":heavy_minus_sign:" * 30)

                # Display relevant documents
                # for index, doc_info in enumerate(relevant_docs, start=1):
                #     st.subheader(f"👉 Document {index}")
                #     st.write("**File** : " + doc_info[0].metadata['name'])

                    # Expander to show more details
                    # with st.expander('Show me 👀'):
                    #     st.info("**Match Score** : " + str(doc_info[1]))
                    #     summary = get_summary(doc_info[0])  # Get summary using LLM
                    #     st.write("**Summary** : " + summary)
                st.success("I hope you found the right candidate.❤️")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# Invoking main function
if __name__ == '__main__':
    main()
