from langchain.document_loaders import PyMuPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import openai
import fitz
import pinecone
def extract_text_from_pdf(pdf_file):
    try:
        pdf_document = fitz.open(pdf_file)
        extracted_text = ""

        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            extracted_text += text

        pdf_document.close()
        return extracted_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
def get_embedding(content, pdf_index):
    openai.api_key = openai_apikey
    response = openai.Embedding.create(
        model = 'text-embedding-ada-002', input=content
    )
    embedding = []
    vec_indexs = []
    index = 0
    for i in response["data"]:
        index += 1
        embedding.append(i["embedding"])
        vec_indexs.append('vec'+ str(index)+'-'+ str(pdf_index))
    return content, embedding, vec_indexs
    

def upserting_to_pinecone(vecs, embeddings, content):
    index_list = pinecone.list_indexes()
    index = pinecone.Index(index_list[0])
    total_vectors = len(vecs)
    batch_size = 50
    
    for i in range(0, total_vectors, batch_size):
        vectors_to_upsert = []
        for j in range(i, min(i+batch_size, total_vectors)):
            vector = {
                'id':vecs[j],
                'values':embeddings[j],
                'metadata':{'content':content[j]}
            }
            vectors_to_upsert.append(vector)
        try:
            print(vector)
            upsert_response = index.upsert(vectors=vectors_to_upsert)
            print('eeeeeeeeeeeeee', upsert_response)
        except Exception as e:
            print('error')
    return 1
def query_pinecone(query):
    sentences, embeddings, vec_indexes = getembedding([query])
    if len(embeddings) == 0:
        return "Creating Embedding Error"
    try:
        query_res = PINECONE_INDEX.query(
            top_k=5,
            include_values=True,
            include_metadata=True,
            vector=embeddings[0],
        )
        return query_res
        # grouped_sentences = {}
        # for result in query_res['matches']:
        #     vector_id = result['id']
        #     file_name = re.search(r"vec\d+-(.+)\.pdf", vector_id).group(1)
        #     print(file_name)
        #     if file_name not in grouped_sentences:
        #         grouped_sentences[file_name] = []
        #     grouped_sentences[file_name].append(result['metadata']['sentence'])

        # return grouped_sentences

    except Exception as e:
        return "Error in Pinecone"
    
def limit_string_tokens(string, max_tokens):
    tokens = string.split()  # Split the string into tokens
    if len(tokens) <= max_tokens:
        return string  # Return the original string if it has fewer or equal tokens than the limit

    # Join the first 'max_tokens' tokens and add an ellipsis at the end
    limited_string = " ".join(tokens[:max_tokens])
    return limited_string


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    speed = 0.05
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.1,
        seed=123,
        # stream=True,
    )
    return completions["choices"][0]["text"]
    # for event in completions:
    #     event_text = event["choices"][0]["text"]
    #     yield event_text

def getembedding(content):
    try:
        apiKey = openai_apikey
        openai.api_key = apiKey
        response = openai.Embedding.create(
            model="text-embedding-ada-002", input=content
        )
        embedding = []
        vec_indexes = []
        index = 0
        for i in response["data"]:
            index += 1
            embedding.append(i["embedding"])
            vec_indexes.append("vec" + str(index))
        return content, embedding, vec_indexes
    except Exception as e:
        # print(traceback.format_exc())
        return [], [], []
    

    
def find_in_pdf(query):
    queryResponse = query_pinecone(query)
    if not queryResponse:
        return('pipecon Error')
    inputSentence = ''
    ids = ''
    for i in queryResponse["matches"]:
        inputSentence += i["metadata"]["content"]
        ids += i["id"]
    inputSentence = limit_string_tokens(inputSentence, 1000)
    print(ids)
    try:
        prompt = f"""
                    You are a chatbot to assist users with your knowledge. You need to give detailed answer about various user queries.
                    You have to use User's language, so for example, if the user asks you something in Dutch, you need to answer in Dutch.
                    You are only a language model, so don't pretend to be a human.
                    Use the next Context to generate answer about user query. If the Context has no relation to user query, you need to generate answer based on the knowledge that you know.
                    And don't mention about the given Context. It is just a reference.
                    Context: {inputSentence}
                    query: {query}

        """

        return {"type": "generic", "content": generate_text(openai_apikey, prompt)}

    except Exception as e:
        return "Net Error"
pinecone.init(api_key='51feb484-4606-4df7-b36b-85c3f827f25f', environment='gcp-starter')
activate_index = pinecone.list_indexes()
PINECONE_INDEX = pinecone.Index(activate_index[0])
openai_apikey = 'sgggggk-cxDMDWQXe8hQ6u3nQAzx" "T3BlbkFJCLfMyBTkUuVcFX7BSPdD'
os.environ['OPENAI_API_KEY'] = 'sggggggk-cxDMDWQXe8hQ6u3nQA" "zxT3BlbkFJCLfMyBTkUuVcFX7BSPdD'
# loader = PyMuPDFLoader('1.pdf')
# document = loader.load()

# text_spliter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 10)
# texts = text_spliter.split_documents(document)


texts = extract_text_from_pdf('1.pdf')
# print(texts)
chunk_size = 1000
overlap = 100
chunks = [
    texts[i:i+chunk_size]
    for i in range(0, len(texts), chunk_size - overlap)
]
print(chunks)


sentence, embedding, vec_index = get_embedding(texts, "file")
print(embedding)
creatEmbedding = upserting_to_pinecone(
    vec_index, embedding, sentence
)
# print(texts[0])
# print(len(texts))
# directory = './storage'
# embeding = OpenAIEmbeddings()
# vect = Chroma.from_documents(documents=texts, embedding=embeding, persist_directory=directory)
# vect.persist()


question  = "what is the name of CEO?"
ans = find_in_pdf(question)
print(ans)