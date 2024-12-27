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
        ...,
        description="Given a user question choose to route it to edit_blog, write_blog or answer",
    )


class BlogWriter:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.2)
        self.crew = CrewClass(llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.5))

        self.memory = ConversationBufferMemory()
        self.blog = {}
        self.router_prompt = """
                            You are a router and your duty is to route the user to the correct expert.
                            Always check conversation history and consider your move based on it.
                            If topic is something about memory, or daily talk route the user to the answer expert.
                            If topic starts something like can u write, or user request you write an article or blog, route the user to the write_blog expert.
                            If topic is user wants to edit anything in the blog, route the user to the edit_blog expert.
                            
                            \nConservation History: {memory}
                            \nTopic: {topic}
                            """

        self.simple_answer_prompt = """
                            You are an expert and you are providing a simple answer to the user's question.
                            
                            \nConversation History: {memory}
                            \nTopic: {topic}
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
        result:  RouteQuery = chain.invoke({"topic": state["topic"], "memory": memory})

        print("Router Result: ", result.way)
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
        self.blog = self.crew.kickoff({"topic": state["topic"]})
        self.memory.save_context(inputs={"input": state["topic"]}, outputs={"output": str(self.blog)})
        pdf_name = generate_pdf(self.blog)
        return {"response": "Here is your blog! ",  "pdf_name": f"{pdf_name}"}

    def edit_blog(self, state: GraphState):
        print("**BLOG EDIT**")
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
        return {"response": "Here is your edited essay! ", "blog": self.blog, "pdf_name": f"{pdf_name}"}
