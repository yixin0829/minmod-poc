import datetime
import os
import re

from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger
from pydantic.v1 import BaseModel
from pydantic.v1.error_wrappers import ValidationError
from tqdm import tqdm

import config.prompts as prompts
from config.config import Config, ExtractionMethod
from schema.mineral_site import (
    BasicInfo,
    DepositTypeCandidates,
    LocationInfo,
    MineralInventory,
    MineralSite,
)
from utils.utils import load_doc, write_model_as_json

# Config loguru logger to log to console and file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(
    os.path.join(Config.LOGGING_DIR, f"minmod_extractor_{timestamp}.log"),
    level=Config.LOGGING_LEVEL,
    rotation="1 week",
    retention="1 month",
)

load_dotenv()


class ExtractorBaseline(object):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
        )

    def extraction_chain_factory(self, output_schema: BaseModel):
        # Create a parser that handles parsing exceptions form PydanticOutputParser by calling LLM to fix the output
        parser = PydanticOutputParser(pydantic_object=output_schema)
        parser_fixing = OutputFixingParser.from_llm(
            llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            parser=parser,
            max_retries=1,
        )

        # Bind the LLM to the response format of the JSON object
        llm = self.llm.bind(response_format={"type": "json_object"})

        # Construct the prompt for the chat model
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.SYS_PROMPT),
                ("human", "{input}"),
            ]
        )

        # Format the prompt partially with the schema of the output model
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        # Under the hood, this is similar to calling create_structured_output_runnable()
        logger.info("Creating baseline structured extraction chain")
        chain = prompt | llm | parser_fixing

        return chain

    def extract_eval(self, inputs: dict, output_schema: BaseModel):
        """
        Extraction wrapper for evaluating the baseline method.

        Args:
            inputs (dict): The inputs to the extraction method. {"input": <doc>}
        """

        # Create baseline runnable
        chain = self.extraction_chain_factory(output_schema)

        # Invoke the chain and return dict
        result = chain.invoke(inputs)
        return {"output": result.dict()}


class ExtractorLLMRetriever(object):
    def __init__(self) -> None:
        pass

    def _llm_retriever_helper(self, query: str, doc: str) -> str:
        """
        Use LLM to retrieve information from the document and return a string containing the retrieved information enclosed in <retrieved></retrieved> XML tags.
        """
        parser_str = StrOutputParser()

        # Construct the prompt for LLM retrieval
        retrieval_prompt = PromptTemplate.from_template(
            prompts.RETRIEVAL_TEMPLATE_STRICT
        )

        chain = retrieval_prompt | self.llm | parser_str

        result = chain.invoke({"query": query, "doc": doc})

        # Extract <retrieved> XML tags from the response
        # Regular expression to match the entire segment including <tag> and </tag>
        pattern = r"<retrieved>((.|\n)*?)</retrieved>"

        # Search for the pattern in the string
        match_xml = re.search(pattern, result)
        match_no_relevant = re.search(r"I can’t find any relevant information", result)

        # Extract and print the content including the tags if a match is found
        if match_xml:
            retrieved_info = match_xml.group(1)
            logger.info(f"Retrieved information: {retrieved_info}")
        elif match_no_relevant:
            logger.info("No relevant information retrieved. Set default value.")
            retrieved_info = "I can’t find any relevant information"
        else:
            logger.warning(
                f'Invalid retrieved information from the response "{result}" Set default value.'
            )
            retrieved_info = "I can’t find any relevant information"

        return retrieved_info

    def _llm_extraction_helper(
        self,
        query: str,
        doc: str,
        retrieved_info: str,
        output_schema: BaseModel,
    ) -> BaseModel:

        parser_pydantic = PydanticOutputParser(pydantic_object=output_schema)
        parser_fixing = OutputFixingParser.from_llm(
            llm=self.llm, parser=parser_pydantic, max_retries=1
        )

        # Bind the LLM to the response format of the JSON object
        llm = self.llm.bind(response_format={"type": "json_object"})

        # * Construct the prompt for LLM extraction (different than baseline)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.SYS_PROMPT),
                ("human", prompts.EXTRACTION_TEMPLATE),
            ]
        )

        # Remove "what's" and "?" from the query
        query_processed = re.sub(r"what's\s|\?", "", query, flags=re.IGNORECASE)
        prompt = prompt.partial(
            format_instructions=parser_pydantic.get_format_instructions()
        )

        # Under the hood, this is similar to calling create_structured_output_runnable()
        chain = prompt | llm | parser_fixing

        logger.info("Invoking the chain to extract info")
        result = chain.invoke(
            {"query": query, "doc": doc, "retrieved_info": retrieved_info}
        )

        return result

    def extract_llm_retriever(self, doc: str, output_schema: MineralSite) -> BaseModel:
        extraction_result = []
        query_schema_pairs = [
            (prompts.BASIC_INFO_QUERY, BasicInfo),
            (prompts.LOCATION_INFO_QUERY, LocationInfo),
            (prompts.MINERAL_INVENTORY_QUERY, MineralInventory),
            (prompts.DEPOSIT_TYPE_QUERY, DepositTypeCandidates),
        ]

        for query, schema in query_schema_pairs:
            logger.info(f"{query=}")

            # Retrieve relevant information
            retrieved_info = self._llm_retriever_helper(query, doc)

            # Extract information
            result_extraction = self._llm_extraction_helper(
                query, doc, retrieved_info, schema
            )

            extraction_result.append(result_extraction)

        try:
            # Construct the MineralSite object
            result = output_schema(
                basic_info=extraction_result[0],
                location_info=extraction_result[1],
                mineral_inventory=extraction_result[2].mineral_inventory,
                deposit_type_candidates=extraction_result[3].candidates,
            )
        except ValidationError as e:
            logger.error(f"Failed to validate the extraction result: {e}")

        return result

    def bulk_extract(self, input_dir: str, method: ExtractionMethod, overwrite: bool):
        """
        Call different extraction methods on documents in the parsed directory.

        Args:
            input_dir (str): The directory containing the parsed documents.
            method (ExtractionMethod): The extraction method to use.
            overwrite (bool): Whether to overwrite existing extraction results.

        Raises:
            ValueError: If the extraction method is not supported.
        """
        if method == ExtractionMethod.BASELINE:
            extraction_func = self.extract_baseline
        elif method == ExtractionMethod.LLM_RETRIEVER:
            extraction_func = self.extract_llm_retriever
        else:
            raise ValueError(f"Extraction method {method} not supported.")

        # Get total file count by counting all .txt files in the parsed directory
        total_file_count = int(
            os.popen(f"find {input_dir} -name '*.txt' | wc -l").read().strip()
        )
        progress_bar = tqdm(total=total_file_count, desc="Extracting Information")
        for root, dirs, files in sorted(os.walk(input_dir)):
            for file in files:
                if file.endswith(".txt"):
                    # Check if file is already extracted in extraction_minmod directory
                    if not overwrite and os.path.exists(
                        f"{self.config.minmod_extraction_dir(method)}/{os.path.splitext(file)[0]}.json"
                    ):
                        logger.info(
                            f"Extraction result for {file} already exists. Skipping."
                        )
                        progress_bar.update(1)
                        continue

                    doc_path = os.path.join(root, file)
                    file_name, doc = load_doc(doc_path)

                    logger.info(f"Extracting information from {doc_path}")
                    result = extraction_func(doc, MineralSite)
                    logger.info(f"Extraction result: {result}")

                    write_model_as_json(
                        result,
                        f"{self.config.minmod_extraction_dir(method)}/{file_name}.json",
                    )

                    progress_bar.update(1)

        progress_bar.close()

    def single_extract(self, file_path: str, method: ExtractionMethod, overwrite: bool):
        """
        Call different extraction methods on a single document.

        Args:
            file_path (str): The file path of the document to extract information from.
            method (ExtractionMethod): The extraction method to use.
            overwrite (bool): Whether to overwrite existing extraction results.

        Raises:
            ValueError: If the extraction method is not supported.
        """
        if method == ExtractionMethod.BASELINE:
            extraction_func = self.extract_baseline
        elif method == ExtractionMethod.LLM_RETRIEVER:
            extraction_func = self.extract_llm_retriever
        else:
            raise ValueError(f"Extraction method {method} not supported.")

        # Check if file is already extracted in extraction_minmod directory
        if (
            os.path.exists(
                f"{self.config.minmod_extraction_dir(method)}/{os.path.splitext(file_path)[0]}.json"
            )
            and not overwrite
        ):
            logger.info(f"Extraction result for {file_path} already exists. Skipping.")
            return

        file_name, doc = load_doc(file_path)

        logger.info(f"Extracting information from {file_path}")
        result = extraction_func(doc, MineralSite)
        logger.info(f"Extraction result: {result}")

        write_model_as_json(
            result,
            f"{self.config.minmod_extraction_dir(method)}/{file_name}.json",
        )


class ExtractorVectorRetriever(object):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
        )

    def retriever_factory(self, doc_name: str):
        # TODO: Debug why the vector retriever is not working in eval (retrieve mixed chunks from other docs)
        with open(os.path.join(Config.PARSED_PDF_DIR_AZURE, doc_name), "r") as f:
            markdown_document = f.read()

        headers_to_split_on = [
            # ("#", "Header 1"),  # In markdown, 2.\ is used for header 1
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)

        embedding_function = OpenAIEmbeddings(model=Config.EMBEDDING_FUNCTION.value)

        vector_db = Chroma.from_documents(
            md_header_splits,
            embedding_function,
            # persist_directory="/home/yixin0829/minmod/minmod-poc/.chroma_db",
        )

        # Creating a retriever from the vector database
        retriever = vector_db.as_retriever(search_type="similarity")

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=ChatOpenAI(temperature=0)
        )
        return retriever_from_llm

    def extraction_chain_factory(self, doc_name: str, output_schema: BaseModel):
        """Construct a runnable for the vector retriever extraction method."""

        # Create a parser that handles parsing exceptions form PydanticOutputParser by calling LLM to fix the output
        parser = PydanticOutputParser(pydantic_object=output_schema)
        parser_fixing = OutputFixingParser.from_llm(
            llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            parser=parser,
            max_retries=1,
        )

        # Bind the LLM to the response format of the JSON object
        llm = self.llm.bind(response_format={"type": "json_object"})

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.SYS_PROMPT_VECTOR_RETRIEVAL),
                ("human", "Relevant sections: {input}"),
            ]
        )
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        retriever = self.retriever_factory(doc_name)
        setup_and_retrieval = RunnableParallel(
            {"input": retriever, "query": RunnablePassthrough()}
        )

        # Creating vector retriever structured extraction chain
        chain = setup_and_retrieval | prompt | llm | parser_fixing

        return chain

    def extract(self, doc_name: str, output_schema: BaseModel) -> BaseModel:
        # Create extraction chains
        chain_basic_info = self.extraction_chain_factory(doc_name, BasicInfo)
        chain_location_info = self.extraction_chain_factory(doc_name, LocationInfo)
        chain_mineral_inventory = self.extraction_chain_factory(
            doc_name, MineralInventory
        )
        chain_deposit_type = self.extraction_chain_factory(
            doc_name, DepositTypeCandidates
        )

        # Invoke the chains for queries to extract different information
        result_basic_info = chain_basic_info.invoke(prompts.BASIC_INFO_QUERY)
        result_location_info = chain_location_info.invoke(prompts.LOCATION_INFO_QUERY)
        result_mineral_inventory: MineralInventory = chain_mineral_inventory.invoke(
            prompts.MINERAL_INVENTORY_QUERY
        )
        result_deposit_type: DepositTypeCandidates = chain_deposit_type.invoke(
            prompts.DEPOSIT_TYPE_QUERY
        )

        # Construct the MineralSite model
        try:
            result = output_schema(
                basic_info=result_basic_info,
                location_info=result_location_info,
                mineral_inventory=result_mineral_inventory.mineral_inventory,
                deposit_type_candidate=result_deposit_type.candidates,
            )
        except ValidationError as e:
            logger.error(
                f"Failed to validate the pydantic extraction result: {e.errors()}"
            )

        return result

    def extract_eval(self, inputs: dict):
        """Wrapper function for evaluation in LangSmith."""
        doc_name = inputs["input"]
        result = self.extract(doc_name + ".md", MineralSite)
        return {"output": result}


if __name__ == "__main__":
    # Extract information from documents in the parsed directory
    config = Config()
    extractor_baseline = ExtractorBaseline(config)
    extractor_vector_retriever = ExtractorVectorRetriever(config)
    print(extractor_vector_retriever.extract("", MineralSite))
