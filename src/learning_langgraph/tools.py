from langchain_core.tools import tool
from learning_langgraph.gmail import send_email
from langchain_tavily import TavilySearch
import requests
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


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
    """
    Get the current weather for a city.

    IMPORTANT: city name should not be abbreviation ,,if the user provide it abbreviated ,  make it full name:

    NY-> new york


    Args:
        city: Name of the city to get the weather for.
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
