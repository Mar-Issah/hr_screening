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
    st.title("HR - Resume Screening Assistance...💁 ")
    st.subheader("I can help you in the resume screening process")

    # Text area for job description
    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here", key="desc", placeholder="Python developer")

    # Text input for number of resumes to return
    document_count = st.text_input("No. of 'RESUMES' to return", key="count", placeholder="3")

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
                # Create a unique ID for this session
                st.session_state['unique_id'] = uuid.uuid4().hex

                # Create a list of documents from uploaded PDF files
                docs = create_docs(pdf, st.session_state['unique_id'])

                # Display the count of uploaded resumes
                st.write("*Resumes uploaded* :" + str(len(docs)))

                # Create embeddings instance
                embeddings = create_embeddings_load_data()

                # Push data to Pinecone
                docsearch = push_to_pinecone(embeddings, docs)

                # Fetch relevant documents from Pinecone
                # relevant_docs = similar_docs(job_description, document_count, "71adf081-aace-4ee4-be84-0a9076ad361e", "gcp-starter", "test", embeddings, st.session_state['unique_id'])
                rel = similarity(job_description, docsearch)
                st.write(rel)

                # Display a line separator
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
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        st.success("Thank you! I hope you found the right candidate.❤️")

# # Invoking main function
if __name__ == '__main__':
    main()
