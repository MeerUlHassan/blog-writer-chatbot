from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal, Dict, Any
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pdf_writer import generate_pdf

from crew import CrewClass, Blog

class GraphState(TypedDict):
    topic: str
    response: str
    documents: List[str]
    blog: Dict[str, Any]
    pdf_name: str

class RouteQuery(BaseModel):
    """Route a user query to direct answer or research."""

    way: Literal["edit_blog","write_blog", "answer"] = Field(
        ..., description="Given a user question choose to route it to edit_blog, write_blog or answer",
    )

class BlogWriter:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.2)
        self.crew = CrewClass(llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.5))

        self.memory = ConversationBufferMemory()
        self.blog = {}
        self.router_prompt = """
                            You are a router and your duty is to route the user to the correct expert.
                            Always check the conversation history and user input.
                            
                            If the user explicitly asks to write a blog but doesn't provide a topic, 
                            respond with "Please provide a topic for the blog." and route them to the answer expert.
                            If the user requests to write a blog and provides a topic, route them to the write_blog expert.
                            If the user wants to edit a blog but hasn't provided an existing blog or specific instructions, 
                            respond with "Please provide the blog content and the specific edits you'd like to make." and route them to the answer expert.
                            For general queries, route them to the answer expert.
                            
                            Conversation History: {memory}
                            User Input: {topic}
                            """

        self.simple_answer_prompt = """
                            You are an expert and you are providing a simple answer to the user's question.
                            
                            Conversation History: {memory}
                            User Input: {topic}
                            """

        builder = StateGraph(GraphState)

        builder.add_node("answer", self.answer)
        builder.add_node("write_blog", self.write_blog)
        builder.add_node("edit_blog", self.edit_blog)

        builder.set_conditional_entry_point(self.router_query,
                                      {"write_blog": "write_blog",
                                       "answer": "answer",
                                       "edit_blog": "edit_blog"})
        builder.add_edge("write_blog", END)
        builder.add_edge("edit_blog", END)
        builder.add_edge("answer", END)

        self.graph = builder.compile()
        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    def router_query(self, state: GraphState):
        print("**ROUTER**")
        prompt = PromptTemplate.from_template(self.router_prompt)
        memory = self.memory.load_memory_variables({})

        router_query = self.model.with_structured_output(RouteQuery)
        chain = prompt | router_query
        result: RouteQuery = chain.invoke({"topic": state["topic"], "memory": memory})

        print("Router Result: ", result.way)
        if result.way == "write_blog" and not state["topic"]:
            return "answer" 
        if result.way == "edit_blog" and (not self.blog or not state["topic"]):
            return "answer" 
        return result.way

    def answer(self, state: GraphState):
        print("**DIRECT ANSWER**")
        prompt = PromptTemplate.from_template(self.simple_answer_prompt)
        memory = self.memory.load_memory_variables({})
        chain = prompt | self.model | StrOutputParser()
        result = chain.invoke({"topic": state["topic"], "memory": memory})

        self.memory.save_context(inputs={"input": state["topic"]}, outputs={"output": result})
        return {"response": result}

    def write_blog(self, state: GraphState):
        print("**BLOG COMPLETION**")
        if not state["topic"]:
            return {"response": "Please provide a topic for the blog."}

        self.blog = self.crew.kickoff({"topic": state["topic"]})
        self.memory.save_context(inputs={"input": state["topic"]}, outputs={"output": str(self.blog)})
        pdf_name = generate_pdf(self.blog)
        return {"response": "Here is your blog! ",  "pdf_name": f"{pdf_name}"}

    def edit_blog(self, state: GraphState):
        print("**BLOG EDIT**")
        if not self.blog:
            return {"response": "No blog content found to edit. Please provide the blog content to proceed."}
        if not state["topic"]:
            return {"response": "Please provide specific edit instructions for the blog."}

        memory = self.memory.load_memory_variables({})
        user_request = state["topic"]
        parser = JsonOutputParser(pydantic_object=Blog)
        prompt = PromptTemplate(
            template=("Edit the Json file as user requested, and return the new Json file."
                     "\n Request:{user_request} "
                     "\n Conservation History: {memory}"
                     "\n Json File: {blog}"
                     " \n{format_instructions}"),
            input_variables=["memory","user_request","blog"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.model | parser
        self.blog = chain.invoke({"user_request": user_request, "memory": memory, "blog": self.blog})

        self.memory.save_context(inputs={"input": state["topic"]}, outputs={"output": str(self.blog)})
        pdf_name = generate_pdf(self.blog)
        return {"response": "Here is your edited blog! ", "blog": self.blog, "pdf_name": f"{pdf_name}"}
