from crewai import Agent
from tools import youtube_channel_tool
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# for open api we use like this
# os.environ["Open_api_key"]=os.getenv("open_api_key")
# os.environ['openai_model']='model_name'

# Blog Researcher Agent
blog_researcher = Agent(
    role='Blog Researcher from YouTube Videos',
    goal='Get relevant video content on the topic "{topic}" from YouTube',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in understanding and analyzing YouTube videos "
        "related to AI, Data Science, and Machine Learning. You extract meaningful insights "
        "and present detailed research summaries for blog writing."
    ),
    tools=[youtube_channel_tool],
    llm=llm,
    allow_delegation=True
)

# Blog Writer Agent
blog_writer = Agent(
    role="Blog Writer",
    goal="Write compelling blog posts based on YouTube videos about '{topic}'",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex tech topics, you create engaging narratives that "
        "educate and captivate readers. You transform raw research into insightful blog content."
    ),
    tools=[youtube_channel_tool],
    llm=llm,
    allow_delegation=False
)
