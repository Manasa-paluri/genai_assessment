import json
import csv
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from dotenv import load_dotenv 
import os 
from bs4 import BeautifulSoup
import re
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
def scrape_website(url):
    
    try:
        service = EdgeService(EdgeChromiumDriverManager().install())
    
        driver = webdriver.Edge(service=service)
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
 
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
 
        # Extract all text content, filter irrelevant parts
        all_text = [p.text for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']) if p.text.strip()]
        cleaned_text = [clean_text(text) for text in all_text if clean_text(text) and not is_irrelevant_text(text)]
 
        return "\n".join(cleaned_text)
 
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""
    finally:
        if 'driver' in locals() and driver:
            driver.quit()


def clean_text(text):
    """ by removing extra whitespace"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(char for char in text if ord(char) >= 32)
    return text


def is_irrelevant_text(text):
    """Checks if text is likely irrelevant (e.g., ads, navigation)."""
    irrelevant_patterns = [
        r'advertisement',
        r'subscribe',
        r'privacy policy',
        r'cookie policy',
        r'navigation',
        r'menu',
        r'social media',
        r'contact us',
        r'legal',
        r'terms of use'
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in irrelevant_patterns)


def extract_website_details(text):
    """Extracts website details using Gemini."""
    genai.configure(api_key=GENAI_API_KEY)  

    model = genai.GenerativeModel('gemini-2.0-pro-exp')  

    prompt = f"""
    Extract the following details about the company from the given text. If any information is not explicitly mentioned, infer it based on the context of the text or general knowledge about the company. Provide concise and meaningful responses in structured JSON format. If any information cannot be found, use your best possible knowledge to infer the data.

    1. Company's mission statement or core values.
    2. Products or services the company offers.
    3. When was the company founded, and who were the founders?
    4. Where is the company's headquarters located?
    5. Who are the key executives or leadership team members?
    6. Has the company received any notable awards or recognitions?

Text: {text}

Please provide the answers in structured JSON format. For any data that is not directly available or explicitly mentioned in the text, infer it to the best of your ability and present it as factual information without indicating that it is inferred like inferred from the knowledge . If you are unable to infer any information, leave the field as "Not Available" .
"""

    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()  
        else:
            return json.dumps({"Error": "No relevant information found."})
    except Exception as e:
        return json.dumps({"Error": str(e)})


def save_to_csv(data, filename="website_details.csv"):
    """Saves the extracted JSON details to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        #  CSV file creation
        fieldnames = ["Website", "Extracted Information"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  #  header

        #  data
        for website, details in data.items():
            writer.writerow({"Website": website, "Extracted Information": details})

    print(f" Data successfully saved to {filename}")



website_urls = [
    "https://www.tcs.com",
    "https://www.starbucks.com",
    "https://www.panasonic.com",
    "https://www.gsk.com",
    "https://www.ford.com",
    "https://www.nespresso.com",
    "https://www.siemens-energy.com",
    "https://www.theheinekencompany.com",
    "https://www.lenovo.com",
    "https://www.americanexpress.com"
]

website_data = {}

# Scrape, extract ,store  data
for url in website_urls:
    print(f"Processing {url}...")
    scraped_text = scrape_website(url)
    if scraped_text:
        details = extract_website_details(scraped_text)
        website_data[url] = details


save_to_csv(website_data)
