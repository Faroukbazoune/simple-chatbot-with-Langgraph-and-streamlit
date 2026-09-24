from langchain_core.tools import tool
from learning_langgraph.gmail import send_email
from langchain_tavily import TavilySearch
import requests
import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

search_tool = TavilySearch(max_result=3)


@tool
def send_gmail_mesage(to: str, body: str, subject: str):
    """this tool is used for sending email message
    Args: to : the person to sent ,
        body : the actuall email ,
        subject : the email subject
    """

    messag_id = send_email(to=to, body=body, subject=subject)

    return f"Email was sent successfully to {to}"


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather conditions for a city.
    Args:
      city: The name of the city. Use the full city name whenever possible. If the user provides a common abbreviation, convert it to the corresponding full city name before using it.

      Examples:
        "NY" -> "New York"
          "LA" -> "Los Angeles"
            "SF" -> "San Francisco"



              Returns: A string containing the current weather information for the city. Important: Do not treat arbitrary short forms as city names. Only expand abbreviations when the intended city is reasonably clear.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather lookup failed: OPENWEATHER_API_KEY is not set."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.HTTPError:
        if response.status_code == 404:
            return f"Could not find weather for '{city}' — check the city name."
        return f"Weather API error: {response.status_code} {response.text}"
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"

    data = response.json()
    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data["wind"]["speed"]

    return (
        f"Weather in {city}:\n"
        f"Temperature: {temperature}°C\n"
        f"Feels like: {feels_like}°C\n"
        f"Humidity: {humidity}%\n"
        f"Conditions: {description}\n"
        f"Wind speed: {wind_speed} m/s"
    )


def ingesting_into_rag(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_db = FAISS.from_documents(chunks, embedding=embedder)
    vector_db.save_local("faiss_db")


def get_retriver():
    vectordb = FAISS.load_local(
        "faiss_db", embeddings=embedder, allow_dangerous_deserialization=True
    )
    retriver = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriver


@tool
def rag_tool(query: str):
    """Search the knowledge base for information relevant to the user's question.

    Use this tool when the user asks about information that may be contained in the connected documents, files, or stored knowledge base. The tool performs semantic search and returns the most relevant pieces of information.

    Always use this tool when answering questions that require information from the knowledge base rather than relying on your own knowledge.

    Args:
    query: A clear, specific search query describing the information you need to find.
    """

    retriver = get_retriver()
    documents = retriver.invoke(query)
    if not documents:
        return "no relevant document found "

    formatted_docs = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        formatted_docs.append(f"""document {index}
source: {source}            
page:{page}
content:{doc.page_content}""")

        return "\n\n".join(formatted_docs)
