from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.tools.bing_search.tool import BingSearchResults as Bing
from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.tools import PubmedQueryRun
from CustomOutputParser import CustomOutputParser
from CustomPromptTemplate import CustomPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.tools import  Tool
from langchain import LLMChain
from langchain.agents import LLMSingleActionAgent
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
#from langchain.output_parsers import StructuredOutputParser as output_parser
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["BING_SUBSCRIPTION_KEY"] = "adf87ddbbf6643948e1c404df7cb0f22"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
pinecone.init(      
	api_key='b1e8aa5b-abdd-4158-9a91-fff36d29aaa3',      
	environment='gcp-starter'      
)      
openai_api_key = "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"

template_with_history = """Answer the following questions as best you can using the knowledge on the data you are trained, but speaking as a biomedical expert. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, use one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a biomedical expert when giving your final answer. Use lots of "Arguments"

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""
llm = ChatOpenAI(model_name = 'gpt-4',temperature=0.7)
search_tool = Bing(api_wrapper=BingSearchAPIWrapper())
embed = OpenAIEmbeddings()
text_field = "text"
index = pinecone.Index('biobot')
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
knowledge_base_tool = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever= vectorstore.as_retriever())
pubmed = PubmedQueryRun()
expanded_tools = [
    Tool(
        name='Knowledge Base',
        func=knowledge_base_tool.run,
        description="Useful when asked about the tools and website of Prepaire labs"
    ),
    Tool(
        name='pubmed',
        func=pubmed.run,
        description="Useful when asked questions in the biomedical domain"),
    Tool(
        name="Search",
        func=search_tool.run,
        description="useful for when you need to answer recent questions"
    )
    
    
]


prompt_with_history = CustomPromptTemplate(
    template= template_with_history,
   tools=expanded_tools,
   input_variables=["input", "intermediate_steps", "history"]
)


output_parser = CustomOutputParser()
# Create the LLM Chain
llm_chain = LLMChain(llm=llm, prompt = prompt_with_history)

# List of tool names
tool_names = [tool.name for tool in expanded_tools]

# Create the custom agent
custom_agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

# Define the memory for the agent
agent_memory = ConversationBufferWindowMemory(k=2)

# Build the Agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=custom_agent, 
    tools=expanded_tools, 
    verbose=True,
    #return_intermediate_steps=True,
    memory=agent_memory
)


