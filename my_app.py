from dotenv import load_dotenv
load_dotenv() 

import gradio as gr
import os
import json
import uuid
import time
import random
import re 
import io
from contextlib import redirect_stdout
from threading import Lock
from typing import Dict, List, Tuple, Any, Generator 
from pathlib import Path
from datetime import datetime
import pytz
import requests
from duckduckgo_search import DDGS 
from newsapi import NewsApiClient
# LangChain components
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun 
# LlamaIndex for Document Summarization
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# Web Browse (Selenium)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException
# Speech Recognition & Text-to-Speech
import speech_recognition as sr
import pyttsx3

# ----------------------
# Configuration
# ----------------------
CONFIG = {
    "llm_model": "gpt-4o-mini",
    "stt_timeout": 5, 
    "history_dir": "chat_histories",
    "system_prompt": """
    You are AIOCO, a highly capable AI assistant with advanced reasoning capabilities. Your traits:
    - Friendly, supportive, proactive, and use emojis often! 😊👍🎉
    - Expert in multitasking with web browsing, image generation, and document analysis.
    - Uses a natural, engaging conversational style with contextual awareness.
    - Verifies sensitive actions and maintains ethical boundaries. Adhere to safety guidelines strictly.
    - Professional yet warm tone with personality. Be conversational!

    **Core Capabilities & Instructions**:

    1. **Web Browse**: Use `web_search` for quick info or `selenium_browse` for dynamic site interaction or accessing specific URLs provided by the user. When calling `selenium_browse`, you MUST set `headless=False` if the input includes '[System Note: User prefers to see the browser when browsing.]' or if the user explicitly requests to see the browser (e.g., 'show the browser', 'open browser visibly', 'make browser visible'). For example, if the input is 'browse to example.com' with the system note, you MUST call `selenium_browse(query='example.com', headless=False)`. If the system note is not present and the user does not request a visible browser, default to `headless=True`. Always let the user know you're browsing! 🌐
    2. **Image Generation**: When asked to create an image, use the `generate_image` tool with the user's description. **Important**: After calling the tool, *only state that the image has been generated and will appear in the gallery*. Do not repeat the success message from the tool (like '[Path: ...]'). Just say something like: "Okay, I've generated the image! You should see it in the gallery shortly. ✨"
    3. **Document Analysis**: If the user uploads a file and asks you to summarize or analyze it, context about the file path will be provided in the input. Use the `summarize_doc` tool with this exact `file_path`. For example, if the input contains "[System Note: User has uploaded a file... path is: '/path/to/doc.pdf']", extract '/path/to/doc.pdf' and call `summarize_doc` with it. 📄➡️📝
    4. **Creative Commons Search**: Use `search_creative_commons` to find CC-licensed images. Mention that the found images will appear in the gallery.
    5. **Weather**: Use `get_weather` for city-specific forecasts. 🌦️
    6. **News**: Use `get_news` for headlines by topic and fetch news.📰
    7. **Calculations**: Use `calculate` for math problems. 🧮
    8. **Date/Time**: Use `get_datetime` for current time info. ⏰
    9. **Todos**: Use `create_todo` to manage tasks. ✅

    **General Guidelines**:
    * Maintain context throughout the conversation. Refer back to previous messages if relevant.
    * Be transparent about what you can and cannot do. If unsure, say so or use web search.
    * Explain complex topics simply.
    * Adapt to the user’s tone. If they are casual, be casual. If formal, be more formal (but always friendly!).
    * When using tools that require API keys, such as weather or news, proceed to use the tool. If the tool indicates that the API key is missing, inform the user they need to set the corresponding environment variable (e.g., `OPENWEATHER_API_KEY` for weather, `NEWS_API_KEY` for news).
    * Always return final answers in user language. but when you want to use tools use English.
    """
}
Path(CONFIG["history_dir"]).mkdir(parents=True, exist_ok=True)
Path("generated_images").mkdir(parents=True, exist_ok=True)
Path("temp_docs").mkdir(parents=True, exist_ok=True) 

# ----------------------
# Custom Tools Implementation
# ----------------------
def web_search(query: str) -> str:
    """Searches the web using DuckDuckGo for current information based on the query."""
    print(f"Tool: web_search called with query: '{query}'")
    try:
        search_engine = DuckDuckGoSearchRun()
        raw_results = search_engine.invoke(query)
        if not raw_results:
            return f"😕 No significant results found for '{query}' on DuckDuckGo."
        summary = f"Web search results for '{query}':\n"
        snippets = raw_results.split("\n")
        summary += "\n".join([f"- {snippet[:1000]}..." for snippet in snippets if snippet][:3]) # Limit length and number
        return summary
    except Exception as e:
        print(f"Error during web search: {str(e)}")
        return f"❌ Error during web search: {str(e)}"

def selenium_browse(query: str, headless: bool = True) -> str:
    print(f"Tool: selenium_browse called with query: '{query}', headless={headless}")
    driver = None
    steps = []
    try:
        options = Options()
        if headless:
            print("DEBUG: Setting Selenium options for HEADLESS mode.")
            options.add_argument("--headless")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-gpu")
        else:
            print("DEBUG: Setting Selenium options for VISIBLE (non-headless) mode.")
            options.add_argument("--start-maximized") 

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        service = Service(ChromeDriverManager().install())
        print("DEBUG: Attempting to start Chrome WebDriver...")
        driver = webdriver.Chrome(service=service, options=options)
        print("DEBUG: WebDriver started successfully.")
        driver.set_page_load_timeout(45)
        driver.set_script_timeout(30)
        driver.header_overrides = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        actions = ActionChains(driver)

        if query.lower().startswith("http://") or query.lower().startswith("https://"):
            url = query
            steps.append(f"🌐 Navigating to {url}")
            driver.get(url)
            time.sleep(random.uniform(3, 6))
            title = driver.title
            steps.append(f"📌 Page Title: {title}")
            body_text = driver.find_element(By.TAG_NAME, "body").text
            content_preview = body_text[:1000].strip()
            if content_preview:
                steps.append(f"📄 Content Preview:\n{content_preview}...")
            else:
                steps.append("📄 No text content extracted from body.")
        else:
            steps.append("🌐 Opening Google...")
            driver.get("https://www.google.com")
            time.sleep(random.uniform(2, 4))
            try:
                consent_buttons = driver.find_elements(By.XPATH, "//button[.//span[contains(text(), 'Accept') or contains(text(), 'Alle akzeptieren')]]")
                if consent_buttons:
                    consent_buttons[0].click()
                    steps.append("✅ Handled cookie consent.")
                    time.sleep(1)
            except Exception:
                steps.append("ℹ️ No obvious cookie consent button found or clickable.")

            search_box = driver.find_element(By.NAME, "q")
            actions.move_to_element(search_box).click().perform()
            steps.append(f"🔍 Searching for '{query}'")
            for char in query:
                search_box.send_keys(char)
                time.sleep(random.uniform(0.05, 0.2))
            search_box.send_keys(Keys.RETURN)
            time.sleep(random.uniform(3, 6))

            page_title = driver.title.lower()
            page_source = driver.page_source.lower()
            if "unusual traffic" in page_source or "recaptcha" in page_source or "before you continue" in page_title:
                steps.append("🛑 Detected potential CAPTCHA or block page. Cannot proceed with search.")
                driver.quit()
                return "\n".join(steps)

            results = driver.find_elements(By.CSS_SELECTOR, "h3")
            if results:
                steps.append("🔎 Top Search Results:")
                for i, result in enumerate(results[:3]):
                    if result.text:
                        steps.append(f"{i+1}. {result.text}")
                try:
                    first_link = results[0].find_element(By.XPATH, "./ancestor::a")
                    if first_link:
                        actions.move_to_element(first_link).click().perform()
                        time.sleep(random.uniform(3, 6))
                        steps.append(f"➡️ Navigated to first result: {driver.title}")
                        body_text = driver.find_element(By.TAG_NAME, "body").text
                        content_preview = body_text[:500].strip()
                        if content_preview:
                            steps.append(f"📄 Content Preview:\n{content_preview}...")
                except Exception as nav_err:
                    steps.append(f"⚠️ Error navigating to first result: {nav_err}")
            else:
                steps.append("😕 No search results found in h3 tags.")

        return "\n".join(steps)

    except TimeoutException:
        steps.append(f"⏱️ Error browsing '{query}': Page load timed out")
        return "\n".join(steps)
    except WebDriverException as e:
        steps.append(f"⚠️ WebDriver error browsing '{query}': {str(e)[:200]}...")
        return "\n".join(steps)
    except Exception as e:
        steps.append(f"❌ Unexpected error browsing '{query}': {str(e)}")
        return "\n".join(steps)
    finally:
        if driver:
            driver.quit()
            print("Browser closed.")

def generate_image(prompt: str) -> str:
    """Generates an image based on a textual prompt using Stability AI."""
    print(f"Tool: generate_image called with prompt: '{prompt}'")
    api_key = os.getenv("STABILITY_AI")
    if not api_key:
        return "⚠️ Image Generation Error: STABILITY_AI API key not set in environment variables."

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"authorization": f"Bearer {api_key}", "accept": "image/*"}
    files = {"none": (None, '')}
    data = {
        "prompt": f"{prompt}, high quality, detailed, cinematic lighting, 4k", 
        "output_format": "webp",
        "style_preset": "digital-art", 
        "aspect_ratio": "16:9" 
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        response.raise_for_status()
        if response.status_code == 200:
            image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"generated_{uuid.uuid4()}.webp"
            image_path = os.path.join(image_dir, image_filename)

            with open(image_path, "wb") as file:
                file.write(response.content)

            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                print(f"Image saved successfully to: {image_path}")
                return f"Image generated successfully. [Path: {image_path}]"
            else:
                print(f"Error: Failed to save image file to {image_path} or file is empty.")
                if os.path.exists(image_path): os.remove(image_path)
                return "❌ Image Generation Error: Image file could not be saved correctly after download."

        elif response.status_code == 400:
             error_detail = response.text
             print(f"Error generating image: HTTP 400 Bad Request. Detail: {error_detail}")
             return f"❌ Image Generation Error: Bad Request (Code 400). Possibly an issue with the prompt or parameters. Details: {error_detail[:200]}"
        elif response.status_code == 401:
             print("Error generating image: HTTP 401 Unauthorized. Check API Key.")
             return "❌ Image Generation Error: Unauthorized (Code 401). Please check your STABILITY_AI API key."
        elif response.status_code == 403:
             print("Error generating image: HTTP 403 Forbidden. Permissions or Org issue.")
             return "❌ Image Generation Error: Forbidden (Code 403). Check permissions or organization settings."
        elif response.status_code == 429:
             print("Error generating image: HTTP 429 Rate Limit Exceeded.")
             return "⚠️ Image Generation Error: Rate limit exceeded (Code 429). Please try again later."
        else:
             error_detail = response.text
             print(f"Error generating image: HTTP {response.status_code}. Detail: {error_detail}")
             return f"❌ Failed to generate image: Server returned HTTP {response.status_code}. Details: {error_detail[:200]}"

    except requests.exceptions.RequestException as e:
        print(f"Network error during image generation: {str(e)}")
        return f"❌ Image Generation Error: Network request failed - {str(e)}"
    except Exception as e:
        print(f"Unexpected error during image generation: {str(e)}")
        return f"❌ Image Generation Error: An unexpected error occurred - {str(e)}"


def summarize_doc(file_path: str) -> str:
    """Summarizes the content of a document given its file path using LlamaIndex."""
    print(f"Tool: summarize_doc called with file_path: '{file_path}'")
    if not os.path.exists(file_path):
        return f"❌ Error summarizing document: File not found at path '{file_path}'."

    try:
        # LlamaIndex setup needs OPENAI_API_KEY implicitly or explicitly
        if not os.getenv("OPENAI_API_KEY"):
             return "❌ Error summarizing document: OPENAI_API_KEY not set for LlamaIndex."

        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        if not documents:
             return f"❌ Error summarizing document: Could not load any content from '{os.path.basename(file_path)}'."

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(response_mode="compact") # Use compact for concise summary

        summary_query = (
            "Provide a concise summary of this document. Include the main topic, key points, "
            "and any significant conclusions or actionable items mentioned. "
            "Format the output clearly, perhaps using bullet points for key takeaways."
        )
        summary_response = query_engine.query(summary_query)

        return f"📄 Document Summary ({os.path.basename(file_path)}):\n\n{str(summary_response)}"

    except ImportError:
         print("LlamaIndex or dependencies not installed.")
         return "❌ Error summarizing document: Required library (LlamaIndex/dependencies) not installed."
    except Exception as e:
        print(f"Error summarizing document '{file_path}': {str(e)}")
        return f"❌ Error summarizing document '{os.path.basename(file_path)}': {str(e)}"

def search_creative_commons(query: str, max_results: int = 10) -> str:
    """Searches for Creative Commons licensed images related to the query using DuckDuckGo."""
    print(f"Tool: search_creative_commons called with query: '{query}', max_results={max_results}")
    try:
        results_list = []
        with DDGS() as ddgs:
            search_keywords = f"{query} creative commons"
            results_generator = ddgs.images(
                keywords=search_keywords,
                region="wt-wt",       # Worldwide
                safesearch="moderate",  
                license_image = "Share" 
            )

            count = 0
            if results_generator:
                 for result in results_generator:
                      if count >= max_results:
                           break
                      if "image" in result and "title" in result and "url" in result:
                           results_list.append(result)
                           count += 1

        if not results_list:
            return f"😕 No Creative Commons images found for '{query}' with 'Share' license. You could try asking for a different license type if needed."

        image_info = []
        for i, result in enumerate(results_list):
             title = result.get('title', 'No title')
             source_url = result.get('url', '#') 
             image_url = result['image']
             info = f"{i+1}. [{title}]({source_url}) - Image URL: {image_url}"
             image_info.append(info)

        text_response = f"🖼️ Found these Creative Commons images for '{query}' (links go to source page):\n"
        text_response += "\n".join(image_info)
        text_response += "\n\Remember to check license terms on the source page before use!"
        return text_response

    except Exception as e:
        print(f"Error finding CC images: {str(e)}")
        return f"❌ Error searching for Creative Commons images: {str(e)}"

def get_weather(city: str) -> str:
    """Gets the current weather and a short forecast for a specified city using OpenWeatherMap."""
    print(f"Tool: get_weather called for city: '{city}'")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "⚠️ Weather Tool Error: OPENWEATHER_API_KEY environment variable is not set."

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    forecast_url = "http://api.openweathermap.org/data/2.5/forecast" # 5 day / 3 hour forecast
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric" # Use metric units (Celsius)
    }

    try:
        current_response = requests.get(base_url, params=params, timeout=10)
        current_response.raise_for_status() 
        current_data = current_response.json()

        if not current_data or current_data.get("cod") != 200:
            return f"😕 Could not retrieve current weather for {city}. Response: {current_data.get('message', 'Unknown error')}"

        weather_desc = current_data["weather"][0]["description"].capitalize()
        temp = current_data["main"]["temp"]
        feels_like = current_data["main"]["feels_like"]
        humidity = current_data["main"]["humidity"]
        wind_speed = current_data["wind"]["speed"] # meters per second
        city_name = current_data["name"] 

        current_weather_str = (
            f"Current weather in {city_name}:\n"
            f"- Conditions: {weather_desc}\n"
            f"- Temperature: {temp}°C (Feels like: {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind: {wind_speed * 3.6:.1f} km/h" # Convert m/s to km/h
        )

        forecast_params = params.copy()
        # cnt=8 means 8 * 3 hours = 24 hours forecast
        forecast_params["cnt"] = 8
        forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        forecast_str = "\n\nForecast (next ~24 hours):"
        if forecast_data.get("cod") == "200" and forecast_data.get("list"):
            # Group by day (simplified) or just list next few entries
            count = 0
            for item in forecast_data["list"]:
                 if count >= 3: break # Show next 3 forecast points (9 hours)
                 dt_obj = datetime.fromtimestamp(item['dt'], tz=pytz.utc).astimezone(pytz.timezone('Europe/Berlin')) # Adjust timezone as needed
                 time_str = dt_obj.strftime('%H:%M')
                 forecast_temp = item['main']['temp']
                 forecast_desc = item['weather'][0]['description'].capitalize()
                 forecast_str += f"\n- {time_str}: {forecast_desc}, {forecast_temp}°C"
                 count +=1
        else:
            forecast_str += "\n- Forecast data currently unavailable."

        return f"🌦️ Weather Report for {city_name}:\n{current_weather_str}\n{forecast_str}"

    except requests.exceptions.HTTPError as http_err:
         if http_err.response.status_code == 404:
              return f"⚠️ Could not find weather data for city: '{city}'. Please check the spelling."
         elif http_err.response.status_code == 401:
              return f"⚠️ Weather Tool Error: Invalid API key (Unauthorized). Please check OPENWEATHER_API_KEY."
         else:
              print(f"HTTP error fetching weather for {city}: {http_err}")
              return f"❌ Error fetching weather data for {city}: {http_err}"
    except requests.exceptions.RequestException as req_err:
         print(f"Network error fetching weather for {city}: {req_err}")
         return f"❌ Network error fetching weather data for {city}: {req_err}"
    except Exception as e:
        print(f"Unexpected error fetching weather for {city}: {str(e)}")
        return f"❌ Unexpected error fetching weather data: {str(e)}"

def calculate(expression: str) -> str:
    """Performs mathematical calculations safely."""
    print(f"Tool: calculate called with expression: '{expression}'")
    try:
        allowed_chars = "0123456789+-*/(). "
        safe_expr = ''.join(c for c in expression if c in allowed_chars)

        if not safe_expr:
             return "❌ Calculation error: Expression is empty after sanitization."
        result = eval(safe_expr, {'__builtins__': None}, {}) 

        return f"🧮 Calculation Result: {expression} = {result}"
    except ZeroDivisionError:
         return "❌ Calculation error: Division by zero is not allowed."
    except SyntaxError:
         return f"❌ Calculation error: Invalid mathematical syntax in '{expression}'."
    except Exception as e:
        print(f"Calculation error for '{expression}': {str(e)}")
        return f"❌ Calculation error: {str(e)}"

def get_datetime(time_zone: str = "Europe/Berlin") -> str:
    """Gets the current date and time for a specified time zone (default: Europe/Berlin)."""
    print(f"Tool: get_datetime called for timezone: '{time_zone}'")
    try:
        tz = pytz.timezone(time_zone)
        now = datetime.now(tz)
        formatted_datetime = now.strftime('%A, %B %d, %Y %I:%M:%S %p %Z (%z)')
        return f"⏰ Current Date and Time in {time_zone}:\n{formatted_datetime}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"❌ Error: Unknown time zone '{time_zone}'. Please provide a valid TZ database name (e.g., 'America/New_York', 'UTC')."
    except Exception as e:
        print(f"Error getting datetime for {time_zone}: {str(e)}")
        return f"❌ Unexpected error getting date and time: {str(e)}"

def create_todo(task: str) -> str:
    """Adds a task to a simple in-memory todo list (for demonstration purposes)."""
    print(f"Tool: create_todo called with task: '{task}'")
    return f"✅ Okay, I've noted down: '{task}'. (This is a demo; the list isn't saved permanently)."

def get_news(query: str = None, sources: str = None, domains: str = None, 
             from_date: str = None, to_date: str = None, language: str = 'en', 
             sort_by: str = 'relevancy', page: int = 1) -> str:
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        return "Error: NEWS_API_KEY is not set. Get one from https://newsapi.org."

    newsapi = NewsApiClient(api_key=api_key)

    try:
        if query:
            # Use get_everything for specific queries
            response = newsapi.get_everything(
                q=query,
                sources=sources,
                domains=domains,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by=sort_by,
                page=page
            )
        else:
            # Use get_top_headlines for general news
            response = newsapi.get_top_headlines(
                sources=sources,
                language=language,
                page=page
            )

        if response['status'] == 'ok':
            articles = response['articles']
            if articles:
                return "\n".join([f"{article['title']} - {article['description'] or 'No description available'}" 
                                 for article in articles[:5]])  # Limit to 5 for brevity
            else:
                return "No news articles found."
        else:
            return f"Error from NewsAPI: {response['message']}"
    except Exception as e:
        return f"Error fetching news: {str(e)}"


# ----------------------
# Core Agent Class
# ----------------------
class AIOCOAgent:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("FATAL ERROR: OPENAI_API_KEY environment variable not set.")

        self.llm = ChatOpenAI(
            model=CONFIG["llm_model"],
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )
        self.current_headless = True  
        def selenium_browse_wrapper(query: str, headless: bool = True):
            effective_headless = self.current_headless if hasattr(self, 'current_headless') else headless
            print(f"Wrapper: Enforcing headless={effective_headless} for query: {query}")
            return selenium_browse(query, effective_headless)

        # Define LangChain Tools
        self.langchain_tools = [
            Tool(name="web_search", func=web_search, description="Search the web for current information, news, facts, or general queries using DuckDuckGo."),
            Tool(name="selenium_browse", func=selenium_browse_wrapper, description="Browse the web dynamically. Use for accessing specific URLs or when standard search isn’t enough. Set headless=False if the input includes '[System Note: User prefers to see the browser]' or user requests to see the browser."),
            Tool(name="generate_image", func=generate_image, description="Generate an image based on a detailed textual prompt. The user provides the description."),
            Tool(name="summarize_doc", func=summarize_doc, description="Summarize the content of a document. Requires the exact 'file_path' of the document as input, which will be provided in the user's message context if a file was uploaded."),
            Tool(name="search_creative_commons", func=search_creative_commons, description="Search for Creative Commons licensed images based on a query. Useful for finding images the user might be able to reuse (license terms must be checked)."),
            Tool(name="get_weather", func=get_weather, description="Get the current weather forecast for a specific 'city'."),
            Tool(name="calculate", func=calculate, description="Perform mathematical calculations based on an 'expression' string."),
            Tool(name="get_datetime", func=get_datetime, description="Get the current date and time, optionally for a specific 'time_zone' (default is Europe/Berlin)."),
            Tool(name="create_todo", func=create_todo, description="Add a 'task' to a temporary user reminder/todo list."),
            Tool(name="get_news", func=get_news, description="Get current news headlines for a specific 'topic' (e.g., technology, business, sports). Defaults to 'general' news in Germany ('de').")
        ]

        # Define the Agent Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONFIG["system_prompt"]),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.agent = create_openai_functions_agent(llm=self.llm, tools=self.langchain_tools, prompt=prompt)
        # Speech Recognition and Synthesis Setup
        self.recognizer = sr.Recognizer()
        self.mic_lock = Lock()
        self.engine = None
        self.engine_lock = Lock()
        self.last_spoken_response = None

        # User Session Management
        self.user_sessions: Dict[str, Dict] = {}

    def _initialize_tts(self):
        """Initializes the pyttsx3 engine if not already done."""
        with self.engine_lock:
            if not self.engine:
                try:
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', 160)
                    self.engine.setProperty('volume', 0.9)
                    print("TTS Engine Initialized.")
                except Exception as e:
                    print(f"Error initializing TTS engine: {e}")
                    self.engine = None

    def _get_user_session(self, user_id: str) -> Dict:
        """Loads or creates a user session, ensuring history and images are properly initialized."""
        if user_id not in self.user_sessions:
            history_file = Path(CONFIG["history_dir"]) / f"{user_id}.json"
            if history_file.exists():
                try:
                    with open(history_file, "r", encoding='utf-8') as f:
                        session_data = json.load(f)
                        history = session_data.get("history", [])
                        if isinstance(history, list) and all(isinstance(item, dict) and "role" in item and "content" in item for item in history):
                            self.user_sessions[user_id] = session_data
                            print(f"Loaded session for user {user_id} from file.")
                        else:
                            print(f"Warning: History format in {history_file} is invalid. Resetting history.")
                            self.user_sessions[user_id] = {
                                "history": [],
                                "preferences": session_data.get("preferences", {}),
                                "created_at": session_data.get("created_at", datetime.now().isoformat()),
                                "images": session_data.get("images", [])  
                            }
                except json.JSONDecodeError:
                    print(f"Error reading history file {history_file}. Resetting history.")
                    self.user_sessions[user_id] = {
                        "history": [],
                        "preferences": {},
                        "created_at": datetime.now().isoformat(),
                        "images": []
                    }
            else:
                print(f"No existing session file found for {user_id}. Creating new session.")
                self.user_sessions[user_id] = {
                    "history": [],
                    "preferences": {},
                    "created_at": datetime.now().isoformat(),
                    "images": []
                }

        session = self.user_sessions[user_id]
        if "history" not in session or not isinstance(session["history"], list):
            session["history"] = []
        if "images" not in session or not isinstance(session["images"], list):
            session["images"] = []

        return session

    def _save_user_session(self, user_id: str):
        """Saves the current user session state."""
        if user_id in self.user_sessions:
            session_data = self.user_sessions[user_id]
            history = session_data.get("history", [])
            if not isinstance(history, list) or not all(isinstance(item, dict) and "role" in item and "content" in item for item in history):
                print(f"Error: Attempting to save history for {user_id} in incorrect format. Aborting save.")
                return

            history_file = Path(CONFIG["history_dir"]) / f"{user_id}.json"
            try:
                with open(history_file, "w", encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2)
                print(f"Session saved successfully for user {user_id} to {history_file}")
            except Exception as e:
                print(f"Error saving session file for user {user_id}: {e}")

    def speak(self, text: str):
        """Speaks the given text using pyttsx3 if voice mode is enabled."""
        if not text or text == self.last_spoken_response:
            return

        self._initialize_tts()
        if self.engine:
            print(f"Speaking: {text[:100]}...")
            with self.engine_lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.last_spoken_response = text
                except Exception as e:
                    print(f"Error during TTS processing: {e}")
                    self.engine = None

    def process_query(self, query: str, user_id: str, voice_enabled: bool = False, uploaded_file: str = None, headless: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Processes a user query and yields responses, including text and images."""
        print(f"\n--- Processing Query ---")
        print(f"User ID: {user_id}")
        print(f"Query: '{query[:100]}...'")
        print(f"Voice Enabled: {voice_enabled}")
        print(f"Uploaded File: {uploaded_file}")
        print(f"Headless Browse: {headless}")

        self.current_headless = headless

        session = self._get_user_session(user_id)
        langchain_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else
            AIMessage(content=msg["content"]) if msg["role"] == "assistant" else
            SystemMessage(content=msg["content"])
            for msg in session["history"]
        ]

        executor = AgentExecutor(
            agent=self.agent,
            tools=self.langchain_tools,
            verbose=True,
            handle_parsing_errors=True
        )

        invoke_input = {"input": query, "chat_history": langchain_history}
        if uploaded_file:
            invoke_input["input"] += f"\n\n[System Note: User has uploaded a file at '{uploaded_file}']"
        if not headless:
            invoke_input["input"] += "\n\n[System Note: User prefers to see the browser when browsing.]"
            invoke_input["headless"] = False 

        final_response = "Sorry, I encountered an issue and couldn't process your request."
        final_images = session["images"]  

        try:
            print("Invoking agent executor...")
            log_capture = io.StringIO()
            with redirect_stdout(log_capture):
                result = executor.invoke(invoke_input)
            final_response = result["output"]
            verbose_output = log_capture.getvalue()

            # Extract image paths from verbose output
            path_matches = re.findall(r"\[Path:\s*(.*?)\s*\]", verbose_output)
            for path in path_matches:
                if os.path.exists(path) and path not in session["images"]:
                    session["images"].append(path)
                    print(f"Added image to session: {path}")
                elif not os.path.exists(path):
                    print(f"Warning: Image path not found: {path}")

            print(f"Final Response: {final_response[:100]}...")
            print(f"Session Images: {session['images']}")

        except Exception as e:
            print(f"❌ Error during agent execution: {e}")
            import traceback
            traceback.print_exc()
            final_response = f"😥 Sorry, an error occurred while I was thinking: {e}"

        if voice_enabled and "error" not in final_response.lower():
            self.speak(final_response)

        yield {"text": final_response, "images": session["images"]}
        print("--- Query Processing Finished ---")
# ----------------------
# Gradio UI Definition
# ----------------------

def create_interface(agent: AIOCOAgent):
    """Creates the Gradio web interface."""

    css = """
    :root {
        --space-gradient: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        --neon-blue: #00b4d8;
        --cyber-pink: #ff006e;
    }

    # .gradio-container {
    #     background: url('background/pexels-pixabay-207529.jpg') !important;
    #     background-size: cover !important;
    #     min-height: 100vh !important;
    #     padding: 20px !important;
    # }

    #chat-container {
        background: rgba(16, 24, 39, 0.9) !important;
        backdrop-filter: blur(12px);
        border: 2px solid var(--neon-blue) !important;
        border-radius: 20px !important;
        box-shadow: 0 0 40px rgba(0, 180, 216, 0.4);
        height: 70vh;
    }

    .message.user {
        background: linear-gradient(135deg, #34c759 0%, #2ecc71 100%) !important;
        border-radius: 25px 25px 5px 25px !important;
        margin-left: 50px !important;
        padding: 15px 25px !important;
        max-width: 80%;
        animation: slideInRight 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    .message.bot {
        background: linear-gradient(135deg, #0984e3 0%, #00b4d8 100%) !important;
        border-radius: 25px 25px 25px 5px !important;
        margin-right: 50px !important;
        animation: slideInLeft 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    #image-gallery {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid var(--neon-blue) !important;
        border-radius: 15px !important;
        padding: 15px !important;
    }

    #submit-button {
        background: linear-gradient(90deg, var(--cyber-pink), #ff477e) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        transition: all 0.3s ease !important;
    }

    #submit-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px #ff006e;
    }
    
    """


    def process_message(message: str, chat_history_list: List[Dict], session_id: str, voice_enabled: bool, visible_Browse: bool, uploaded_file_obj: Any) -> Generator[Tuple[List[Dict], str, List[str]], None, None]:
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_history_list = []

        file_path_to_send = None
        user_content = message
        if uploaded_file_obj:
            file_path_to_send = uploaded_file_obj.name
            user_content += f"\n[User uploaded file: {os.path.basename(file_path_to_send)}]"

        chat_history_list.append({"role": "user", "content": user_content})

        try:
            response_generator = agent.process_query(message, session_id, voice_enabled, file_path_to_send, not visible_Browse)
            for response_chunk in response_generator:
                final_response_text = response_chunk["text"]
                final_images = response_chunk["images"]  
                break  

            chat_history_list.append({"role": "assistant", "content": final_response_text})
            print(f"Text processed. Response: '{final_response_text[:50]}...'. Images: {final_images}")

            session = agent._get_user_session(session_id)
            session["history"] = chat_history_list.copy()
            agent._save_user_session(session_id)

            yield chat_history_list, session_id, final_images  

        except Exception as e:
            print(f"Error in process_message callback: {e}")
            error_message = f"❌ Sorry, an internal error occurred: {e}"
            chat_history_list.append({"role": "assistant", "content": error_message})
            session = agent._get_user_session(session_id)
            session["history"] = chat_history_list.copy()
            agent._save_user_session(session_id)
            yield chat_history_list, session_id, session["images"] if "images" in session else []

    def handle_audio(audio_filepath: str, chat_history_list: List[Dict], session_id: str, voice_enabled: bool, visible_Browse: bool) -> Generator[Tuple[List[Dict], str, List[str]], None, None]:
        """Handles audio input recording/upload."""
        print(f"Handling audio file: {audio_filepath}")
        if not audio_filepath:
            print("Audio filepath is None, skipping.")
            yield chat_history_list, session_id, []
            return

        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"New session started from audio: {session_id}")
            chat_history_list = [] 

        transcript = ""
        try:
            with sr.AudioFile(audio_filepath) as source:
                audio_data = agent.recognizer.record(source)
                google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            try:
                if google_credentials:
                    print("Using Google Cloud Speech Recognition with Persian (fa-IR)")
                    transcript = agent.recognizer.recognize_google_cloud(audio_data, credentials_json=google_credentials, language='fa-IR')
                else:
                    print("Using Google Web Speech API with Persian (fa-IR)")
                    transcript = agent.recognizer.recognize_google(audio_data, language='fa-IR')
                print(f"Transcript (Persian): {transcript}")
            except sr.UnknownValueError:
                print("Could not understand audio in Persian, trying English...")
                # Fall back to English (en-US)
                try:
                    if google_credentials:
                        print("Using Google Cloud Speech Recognition with English (en-US)")
                        transcript = agent.recognizer.recognize_google_cloud(audio_data, credentials_json=google_credentials, language='en-US')
                    else:
                        print("Using Google Web Speech API with English (en-US)")
                        transcript = agent.recognizer.recognize_google(audio_data, language='en-US')
                    print(f"Transcript (English): {transcript}")
                except sr.UnknownValueError:
                    print("Speech Recognition could not understand audio in English either")
                chat_history_list.append({"role": "assistant", "content": "🎤 I didn't catch that. Could you please repeat?"})
                yield chat_history_list, session_id, []
                return

            chat_history_list.append({"role": "user", "content": transcript})

            final_response_text = ""
            final_images = []
            response_generator = agent.process_query(transcript, session_id, voice_enabled, None, not visible_Browse)
            for response_chunk in response_generator:
                final_response_text = response_chunk["text"]
                final_images = response_chunk["images"]

            chat_history_list.append({"role": "assistant", "content": final_response_text})

            print(f"Audio processed. Response: '{final_response_text[:50]}...'. Images: {final_images}")

            agent._save_user_session(session_id)
            yield chat_history_list, session_id, final_images

        except FileNotFoundError:
            error_msg = f"Error: Audio file not found at {audio_filepath}"
            print(error_msg)
            chat_history_list.append({"role": "assistant", "content": f"❌ Internal Error: {error_msg}"})
            yield chat_history_list, session_id, []
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            chat_history_list.append({"role": "assistant", "content": f"❌ An internal error occurred while processing audio: {e}"})
            agent._save_user_session(session_id)
            yield chat_history_list, session_id, []
        finally:
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    os.remove(audio_filepath)
                    print(f"Cleaned up temporary audio file: {audio_filepath}")
                except OSError as e:
                    print(f"Warning: Error deleting temporary audio file {audio_filepath}: {e}")


    def new_chat(session_id: str) -> Tuple[List[Dict], str, List[str]]:
        """Clears the chat history and starts a new session."""
        print(f"Starting new chat. Old Session ID: {session_id}")
        new_session_id = str(uuid.uuid4())
        session = agent._get_user_session(new_session_id) 
        return [], new_session_id, session["images"] 

    #Build Gradio Interface 
    with gr.Blocks(css=css, title="AIOCO Assistant", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink")) as demo:
        session_id = gr.State(value=str(uuid.uuid4()))

        with gr.Row(): 
            with gr.Column(scale=3):
                gr.Markdown("## 🚀 AIOCO - Your AI Companion 💫", elem_id="title")
                avatar_user = "user.png" #User icon
                avatar_bot = "bot.png" #Bot icon

                chatbot = gr.Chatbot(
                    [],
                    elem_id="chat-container",
                    avatar_images=(avatar_user, avatar_bot),
                    height=600,
                    scale=5,
                    feedback_options=["Like","Dislike"],
                    type='messages', 
                    layout="bubble"
                 )

                with gr.Row(elem_id="input-row"):
                    mic_input = gr.Audio(
                         sources=["microphone"],
                         type="filepath", 
                         label="Talk To Aioco",
                         show_label=False,
                         editable=False,
                         elem_id="mic-input",
                         scale=0 
                    )
                    textbox = gr.Textbox(
                        placeholder="Type or record your message...",
                        show_label=False,
                        container=False, 
                        scale=5,
                        lines=2, 
                        elem_id="textbox"
                    )
                    submit_btn = gr.Button("Send", scale=1, elem_id="submit-button")

                with gr.Row():
                     voice_enabled = gr.Checkbox(label="🔊 Speak Response(en)", value=False, elem_classes="gradio-checkbox")
                     visible_Browse = gr.Checkbox(label="👁️ Show Browser", value=False, elem_classes="gradio-checkbox") # Default to headless
                     new_chat_btn = gr.Button("✨ New Chat", elem_id="new-chat-button")

            with gr.Column(scale=1):
                 uploaded_file = gr.File(
                     label="📎 Upload Document (PDF, DOCX, TXT)",
                     file_types=[".pdf", ".docx", ".txt"],
                     elem_id="file-upload" 
                 )
                 gallery = gr.Gallery(
                     label="🖼️ Image Gallery",
                     elem_id="image-gallery",
                     columns=2,
                     object_fit="contain",
                     height=400,
                     preview=True 
                  )
                 gr.Markdown("#Quick Tools#:\n- **Browse**: Browse the web dynamically.\n- **Image Generation**: Generate images by prompts.\n- **Weather**: Get current weather forecasts.\n- **Creative Commons**:search for images online\n- **News**: Get current news headlines.")


        # --- Event Listeners ---
        common_inputs = [chatbot, session_id, voice_enabled, visible_Browse]
        common_outputs = [chatbot, session_id, gallery]

        textbox.submit(
             process_message,
             inputs=[textbox, *common_inputs, uploaded_file],
             outputs=common_outputs
        ).then(lambda: gr.update(value=None), inputs=None, outputs=textbox, queue=False) # Clear textbox after submit

        submit_btn.click(
            process_message,
            inputs=[textbox, *common_inputs, uploaded_file],
            outputs=common_outputs
        ).then(lambda: gr.update(value=None), inputs=None, outputs=textbox, queue=False) # Clear textbox after click

        mic_input.stop_recording(
             handle_audio,
             inputs=[mic_input, *common_inputs],
             outputs=common_outputs
        )

        new_chat_btn.click(
             new_chat,
             inputs=[session_id],
             outputs=common_outputs, 
             queue=False
        )
        demo.load(
            None,
            None,
            None,
            js="""
            () => {
                // Add cosmic particles
                const particles = document.createElement('div');
                particles.style.position = 'fixed';
                particles.style.top = '0';
                particles.style.left = '0';
                particles.style.width = '100%';
                particles.style.height = '100%';
                particles.style.pointerEvents = 'none';
                document.body.appendChild(particles);
                
                for(let i=0; i<100; i++) {
                    const star = document.createElement('div');
                    star.style.position = 'absolute';
                    star.style.width = Math.random()*3 + 'px';
                    star.style.height = star.style.width;
                    star.style.background = '#fff';
                    star.style.borderRadius = '50%';
                    star.style.left = Math.random()*100 + '%';
                    star.style.top = Math.random()*100 + '%';
                    star.style.animation = `twinkle ${2+Math.random()*3}s infinite`;
                    particles.appendChild(star);
                }
                
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes twinkle {
                        0% { opacity: 0; transform: scale(0); }
                        50% { opacity: 1; transform: scale(1); }
                        100% { opacity: 0; transform: scale(0); }
                    }
                `;
                document.head.appendChild(style);
            }
            """
        )

    return demo

# ----------------------
# Main Execution Block
# ----------------------
if __name__ == "__main__":
    print("Starting AIOCO Agent...")
    try:
        agent = AIOCOAgent()
        print("Agent initialized successfully.")
        demo = create_interface(agent)
        print("Gradio interface created. Launching...")
        demo.launch(share=True)
    except Exception as e:
        print(f"\n--- FATAL ERROR DURING STARTUP ---")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("------------------------------------")
        print("Please check error messages, API keys, and dependencies.")
