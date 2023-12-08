import os
from utils import get_config
from manager import OpenAIManager
from langsmith.run_helpers import traceable


os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = get_config("KEY_LANGSMITH")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "daily papers"

manager = OpenAIManager()


@traceable(run_type="chain", name="Search Papers Chain")
def search_papers_chain():
    url = "https://huggingface.co/papers"

import requests
from bs4 import BeautifulSoup

def get_papers_from_huggingface():
    url = 'https://huggingface.co/papers'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    papers = soup.find_all('a', class_='paper-card')
    papers_info = []

    for paper in papers:
        title = paper.find('h4').text.strip()
        link = paper.get('href', '')
        full_link = f"https://huggingface.co{link}"
        papers_info.append({'title': title, 'url': full_link})

    return papers_info

# 使用这个函数来获取论文信息
papers_info = get_papers_from_huggingface()

# 打印结果
for info in papers_info:
    print(f"Title: {info['title']}, URL: {info['url']}")