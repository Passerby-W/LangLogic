import argparse
import base64
import configparser
import datetime
import json
import os
import re
from collections import namedtuple

import arxiv
import numpy as np
import openai
import requests
import tenacity
import tiktoken

import fitz, io, os
from PIL import Image
from bs4 import BeautifulSoup


class Paper:
    def __init__(self, path, title="", url="", abstraction="", authors=None):
        self.path = path  # pdf路径

        self.url = url  # 文章链接
        self.section_names = []  # 段落标题
        self.section_texts = {}  # 段落内容
        self.abstraction = abstraction
        self.title_page = 0
        if authors is None:
            authors = []
        if title == '':
            self.pdf = fitz.open(self.path)  # pdf文档
            self.title = self.get_title()
            self.parse_pdf()
        else:
            self.title = title
        self.authors = authors
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]


def get_papers_from_huggingface_daily_papers():
    papers_info = []
    url = 'https://huggingface.co/papers'
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    section_content = soup.find('section', class_='container relative mb-20 mt-8 md:mt-14')
    cursor_pointer_links = section_content.find_all('a', class_='cursor-pointer')

    for link in cursor_pointer_links:
        arxiv_id = link.get('href').split("/")[-1]
        title = link.text.strip()
        if arxiv_id and title:
            papers_info.append({"title": title, "arxiv_id": arxiv_id})
    return papers_info


print(get_papers_from_huggingface_daily_papers())
