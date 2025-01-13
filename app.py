
import embedding as em
import streamlit as st
import PyPDF2

st.title("Retrival Augmented Generation Application using ChromaDb and OpenAI")
st.write("upload your book and ask a query")

#-----------------------**************----------------------------------
#                          Sidebar code

st.sidebar.title("Menu")
n_results=st.sidebar.number_input("select no. of results",min_value=1,max_value=5,value=1)

#----------------------------************--------------------------------
#                            main page code

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    # Extract file content as text
    if uploaded_file.name.endswith(".txt"):
        # For text files
        file_content = uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_content = " ".join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        st.stop()

    st.write("File upload successfull:")

    file_name = uploaded_file.name
    collection= em.create_embeddings(file_content,file_name)
    print(collection)


user_input = st.text_input("Ask me something:")
if user_input:
    if st.button("Get Answers"):
        
        ans=em.get_final_answer(user_input,file_content,collection,n_results,file_name)
        st.write(ans)
        





