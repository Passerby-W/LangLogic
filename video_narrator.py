from llm import OpenAILLM
import os
from utils import get_config
from manager import OpenAIManager
from langsmith.run_helpers import traceable
import base64
import cv2

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = get_config("KEY_LANGSMITH")
os.environ["LANGCHAIN_PROJECT"] = "test"

manager = OpenAIManager()
llm = OpenAILLM()


@traceable(run_type="llm", name="openai.chat.ChatCompletion.create")
def chat_model(*args, **kwargs):
    return manager.chat(*args, **kwargs)


@traceable(run_type="tool")
def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    return video


@traceable(run_type="tool")
def frame_video(video):
    base64_frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return base64_frames


@traceable(run_type="chain")
def preprocessing_chain(video_path):
    video = load_video(video_path)
    base64_frames = frame_video(video)
    return base64_frames


@traceable(run_type="tool")
def prepare_data(base64_frames):
    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    "对下面我想上传的视频中的帧，生成一段吸引人的叙述。",
                    *map(lambda x: {"image": x, "resize": 768}, base64_frames[0::50]),
                ],
            },
        ],
        "max_tokens": 200
    }
    return data


@traceable(run_type="tool")
def get_ans(completion):
    return completion.choices[0].message.content


@traceable(run_type="chain")
def narrator_chain(base64_frames):
    data = prepare_data(base64_frames)
    completion = chat_model(**data)
    ans = get_ans(completion)
    return ans


@traceable(run_type="chain")
def main_chain(video_path):
    base64_frames = preprocessing_chain(video_path)
    ans = narrator_chain(base64_frames)
    return ans


a = main_chain("test_data/test.mp4")
print(a)

