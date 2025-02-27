from langgraph.graph import StateGraph, START, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from src.components.sql_history import SQLChatHistory
import os
import yaml

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

with open('src\config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)


class GradeResult(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class Chatbot:
    def __init__(self, retriever):
        self.llm = ChatGroq(model=config['config']['llm'])
        self.retriever = retriever
        self.sql = SQLChatHistory()  # Pass user_id here
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="retrieve",
            description=config['prompts']['retriever']
        )
        self.user_id = config['config']['user_id']

    def retrieve_tool(self, state: AgentState):
        question = state['messages'][-1].content 
        result = self.retriever_tool.invoke({"query": question})

        self.sql.save_message("user", question)

        return {"messages": [result]}

    def process_query(self, state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        question = last_message.content

        chat_history = self.sql.get_chat_history()

        prompt = PromptTemplate(
            template=config['prompts']['process_query'],
            input_variables=["question", 'chat_history']
        )
        chain = prompt | self.llm
        print("process_query")
        response = chain.invoke({"question": question, 'chat_history': chat_history})

        self.sql.save_message("assistant", response.content)

        return {"messages": [response]}

    def grade_documents(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content

        prompt = PromptTemplate(
            template="""You are a grader deciding if a document is relevant to a user’s question.
                        Here is the document: {context}
                        Here is the user’s question: {question}
                        Answer 'yes' or 'no'.""",
            input_variables=["context", "question"]
        )

        chain = prompt | self.llm.with_structured_output(GradeResult)
        print('grad_document')
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        return "Output_Generator" if score == "yes" else "Query_Rewriter"

    def generate_output(self, state: AgentState):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        chat_history = self.sql.get_chat_history()

        prompt = PromptTemplate(
            template=config['prompts']['generate_output'],
            input_variables=["customer_question", "retrieved_context", 'chat_history']
        )

        chain = prompt | self.llm
        print('generate_output')

        response = chain.invoke({"retrieved_context": docs, "customer_question": question, 'chat_history': chat_history})

        self.sql.save_message("assistant", response.content)

        return {"messages": [response]}

    def rewrite_query(self, state: AgentState):
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content
        print(question)
        return {"messages": [question]}

    def __call__(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("My_Ai_Assistant", self.process_query)
        workflow.add_node("Vector_Retriever", self.retrieve_tool)
        workflow.add_node("Output_Generator", self.generate_output)
        workflow.add_node("Query_Rewriter", self.rewrite_query)

        workflow.add_edge(START, "Vector_Retriever")

        workflow.add_conditional_edges("Vector_Retriever",
                                      self.grade_documents,
                                      {"Output_Generator": "Output_Generator",
                                       "Query_Rewriter": "Query_Rewriter"
                                       }
                                      )

        workflow.add_edge("Output_Generator", END)
        workflow.add_edge("Query_Rewriter", "My_Ai_Assistant")
        workflow.add_edge("My_Ai_Assistant", END)
        self.app1 = workflow.compile()
        return self.app1