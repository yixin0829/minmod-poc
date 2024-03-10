from langchain_benchmarks import clone_public_dataset, registry
from loguru import logger

task = registry["Email Extraction"]
logger.info(f"Cloning {task.name} dataset")
logger.info(f"Task description {task.description}")

clone_public_dataset(task.dataset_id, dataset_name=task.name)

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0).bind_functions(
    functions=[task.schema],
    function_call=task.schema.schema()["title"],
)

output_parser = JsonOutputFunctionsParser()
extraction_chain = task.instructions | llm | output_parser | (lambda x: {"output": x})

from langchain_benchmarks.extraction import get_eval_config
from langsmith.client import Client
