from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List, TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()

class Paragraph(TypedDict):
    sub_header: str
    paragraph: str

class Blog(BaseModel):
    header: str = Field(..., description="The header of the blog")
    entry: str = Field(..., description="The entry of the blog")
    paragraphs: List[Paragraph] = Field(..., description="The paragraphs of the blog")
    conclusion: str = Field(..., description="The conclusion of the blog")
    seo_keywords: List[str] = Field(..., description="The SEO keywords of the blog")

class CrewClass:
    """Blog Writing Crew Class"""
    def __init__(self, llm):
        self.llm = llm

        self.researcher = Agent(
        role="Content Researcher",
        goal="Research accurate and up-to-date content on {topic} for a blog.",
        backstory="You're researching content to help create a blog about the topic: {topic}. "
              "You gather information that is engaging, informative, and tailored to the blog's audience, helping them learn and stay engaged. "
              "Your research serves as the foundation for the Content Writer to craft a compelling blog post on this topic.",
            verbose=True,
            llm=self.llm,
        )

        self.writer = Agent(
            role="Content Writer",
            goal="Write insightful and factually accurate "
                 "opinion piece about the provided topic",
            backstory="You're working on a writing a new opinion piece about the provided topic."
                      "You base your writing on the work of the Content Researcher, who provides an outline and relevant context about the topic."
                      "You follow the main objectives and direction of the outline, as provide by the Content Researcher."
                      "You also provide objective and impartial insights and back them up with information provide by the Content Researcher.",
            verbose=True,
            llm=self.llm
        )
        self.editor = Agent(
            role="Content Editor",
            goal="Edit a given essay to align with the writing style of the organization.",
            backstory="You are an editor who receives a blog from the Content Writer."
                      "Your goal is to review the blog to ensure that it follows best practices, provides balanced viewpoints"
                      "when providing opinions or assertions,and also avoids major controversial topics or opinions when possible.",
            verbose=True,
            llm=self.llm
        )

        self.research = Task(
            description=(
                "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
                "2. Identify the target audience, considering their interests and pain points.\n"
                "3. Research a detailed content outline including an introduction, key points, and a conclusion.\n"
                "4. Include SEO keywords and relevant data or sources."
            ),
            expected_output="A comprehensive document with an outline, audience analysis, SEO keywords, and resources.",
            tools = [search_tool],
            agent=self.researcher,
        )

        self.write = Task(
            description=(
                "1. Use the content to craft a compelling blog.\n"
                "2. Incorporate SEO keywords naturally.\n"
                "3. Sections/Subtitles are properly named in an engaging manner.\n"
                "4. Ensure the blog is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n"
                "5. Proofread for grammatical errors and alignment with the brand's voice.\n"
                "6. Pick a suitable header\n"
            ),
            expected_output="A well-written blog in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
            context=[self.research],
            agent=self.writer,
        )

        self.edit = Task(
            description="Proofread the given blog for grammatical errors and alignment with the brand's voice.",
            expected_output="A well-written blog in required format, ready for publication, each section must have 2 or 3 paragraphs.",
            output_json = Blog,
            context=[self.write],
            agent=self.editor
        )

    def kickoff(self,*args):
        return Crew(
            agents=[self.researcher, self.writer, self.editor],
            tasks=[self.research, self.write, self.edit],
            process=Process.sequential,
            verbose=True,
            memory=True
        ).kickoff(*args)