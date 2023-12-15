import os
import requests
from bs4 import BeautifulSoup
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
    papers_info = get_papers_from_huggingface()


def get_papers_from_huggingface():
    papers_info = []
    url = 'https://huggingface.co/papers'
    response = requests.get(url)

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有 class 为 'cursor-pointer' 的 <a> 元素
    section_content = soup.find('section', class_='container relative mb-20 mt-8 md:mt-14')
    cursor_pointer_links = section_content.find_all('a', class_='cursor-pointer')

    # 提取每个元素的 href 属性和文本内容
    for link in cursor_pointer_links:
        href = link.get('href')
        title = link.text.strip()
        if href and title:
            info_url = f"https://huggingface.co{href}"
            response = requests.get(info_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', class_='btn inline-flex h-9 items-center')

            for source_link in links:
                if source_link.text.strip() == 'View PDF':
                    info = {"title": title, "url": source_link.get('href')}
                    papers_info.append(info)

    return papers_info


@traceable(run_type="chain", name="Search Papers Chain")
def read_papers_chain():
    pass