from crewai import Task
from tools import youtube_channel_tool
from agent import blog_researcher, blog_writer

# Research Task
research_task = Task(
    description=(
        "Identify the video about the topic '{topic}'. "
        "Get detailed information about the video from the YouTube channel."
    ),
    expected_output=(
        "A comparative 3-paragraph report on the topic '{topic}', "
        "based on the content of the video(s) found on YouTube."
    ),
    tools=[youtube_channel_tool],
    agent=blog_researcher,
)

# Writing Task
write_task = Task(
    description=(
        "Get the information from the YouTube channel on the topic '{topic}' "
        "and use it to write a blog post."
    ),
    expected_output=(
        "Summarize the information from the YouTube video(s) on the topic '{topic}' "
        "and create blog content."
    ),
    tools=[youtube_channel_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md'
)
