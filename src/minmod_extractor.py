import datetime
import json
import os
import re

import tqdm
from dotenv import load_dotenv
from langchain.chains import create_structured_output_runnable
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic.v1 import BaseModel
from pydantic.v1.error_wrappers import ValidationError

import prompts as prompts
from config import Config, ExtractionMethod
from schema import (
    BasicInfo,
    DepositTypeCandidates,
    LocationInfo,
    MineralInventory,
    MineralSite,
)

# Config loguru logger to log to console and file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(
    os.path.join(Config.LOGGING_DIR, f"minmod_extractor_{timestamp}.log"),
    level=Config.LOGGING_LEVEL,
    rotation="1 week",
    retention="1 month",
)

load_dotenv()


class MinModExtractor(object):
    def __init__(self, MODEL_NAME: str) -> None:
        self.config = Config()
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0.5, max_tokens=2048)

    def _load_doc(self, doc_path: str) -> tuple[str, str]:
        """
        Load the document from the given file path.

        Args:
            doc_path (str): The path to the document file.

        Returns:
            tuple[str, str]: A tuple containing the file name and the content of the document.
        """
        logger.info(f"Loading document from {doc_path}")
        with open(doc_path, "r") as f:
            doc = f.read()
        file_name = os.path.splitext(os.path.basename(doc_path))[0]
        logger.info(f"File name: {file_name}")
        return file_name, doc

    def _write_model_as_json(self, model: BaseModel, file_path: str):
        """
        Write a extracted Pydantic model object as a JSON file to the given file path.
        """
        logger.info("Writing the extraction result to a JSON file")
        # Check if the file directory exists, if not, create it.
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(file_path, "w") as f:
            f.write(model.json(indent=4))

        logger.info(f"Saved at {file_path}")

    def extract_baseline(self, doc_path: str, output_schema: BaseModel) -> MineralSite:
        file_name, doc = self._load_doc(doc_path)

        # Create a parser that handles parsing exceptions form PydanticOutputParser by calling LLM to fix the output
        parser_pydantic = PydanticOutputParser(pydantic_object=output_schema)
        parser_fixing = OutputFixingParser.from_llm(
            llm=self.llm, parser=parser_pydantic, max_retries=1
        )

        # Bind the LLM to the response format of the JSON object
        llm = self.llm.bind(response_format={"type": "json_object"})

        # Construct the prompt for the chat model
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.sys_prompt),
                ("human", "{input}"),
            ]
        )

        # Format the prompt partially with the schema of the output model
        prompt = prompt.partial(
            format_instructions=parser_pydantic.get_format_instructions()
        )

        # Under the hood, this is similar to calling create_structured_output_runnable()
        logger.info("Creating structured output chain")
        chain = prompt | llm | parser_fixing

        logger.info("Invoking the chain to process single doc")
        result = chain.invoke({"input": doc})

        self._write_model_as_json(
            result,
            f"{self.config.minmod_extraction_dir(ExtractionMethod.BASELINE)}/{file_name}.json",
        )

        return result

    def _llm_retrieval_helper(self, query: str, doc: str) -> str:
        """
        Use LLM to retrieve information from the document and return a string containing the retrieved information enclosed in <retrieved></retrieved> XML tags.
        """
        parser_str = StrOutputParser()

        # Construct the prompt for LLM retrieval
        retrieval_prompt = PromptTemplate.from_template(
            prompts.retrieval_template_strict
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
            # TODO: Resolve invalid response from LLM due to copy right issue
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
                ("system", prompts.sys_prompt),
                ("human", prompts.extraction_template),
            ]
        )

        # Remove "what's" and "?" from the query
        query_processed = re.sub(r"what's\s|\?", "", query, flags=re.IGNORECASE)
        prompt = prompt.partial(
            format_instructions=parser_pydantic.get_format_instructions()
        )

        # Under the hood, this is similar to calling create_structured_output_runnable()
        chain = prompt | llm | parser_fixing

        logger.info("Invoking the chain to extract information")
        result = chain.invoke(
            {"query": query, "doc": doc, "retrieved_info": retrieved_info}
        )

        return result

    def extract_llm_retrieval(
        self, doc_path: str, output_schema: MineralSite
    ) -> MineralSite:
        file_name, doc = self._load_doc(doc_path)

        extraction_result = []
        query_schema_pairs = [
            (prompts.basic_info_query, BasicInfo),
            (prompts.location_info_query, LocationInfo),
            (prompts.mineral_inventory_query, MineralInventory),
            (prompts.deposit_type_query, DepositTypeCandidates),
        ]

        for query, schema in query_schema_pairs:
            logger.info(f"{query=}")

            # Retrieve relevant information
            retrieved_info = self._llm_retrieval_helper(query, doc)

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
                mineral_inventory=extraction_result[2],
                deposit_type_candidates=extraction_result[3],
            )

            self._write_model_as_json(
                result,
                f"{self.config.minmod_extraction_dir(ExtractionMethod.LLM_RETRIEVAL)}/{file_name}.json",
            )
        except ValidationError as e:
            # Write unvalidated JSON as a fallback
            logger.warning(
                f"Failed to validate the extraction result, write unvalidated JSON as a fallback: {e}"
            )

            extraction_result = [v.dict() for v in extraction_result]
            result = {}
            for d in extraction_result:
                result.update(d)

            with open(
                f"{self.config.minmod_extraction_dir(ExtractionMethod.LLM_RETRIEVAL)}/{file_name}_unvalidated.json",
                "w",
            ) as f:
                json.dump(result, f, indent=4)

        return result

    def bulk_extract(self, method: ExtractionMethod, overwrite: bool):
        """
        Call different extraction methods on documents in the parsed directory.

        Args:
            method (ExtractionMethod): The extraction method to use.
            overwrite (bool): Whether to overwrite existing extraction results.

        Raises:
            ValueError: If the extraction method is not supported.
        """
        if method == ExtractionMethod.BASELINE:
            extraction_method = self.extract_baseline
        elif method == ExtractionMethod.LLM_RETRIEVAL:
            extraction_method = self.extract_llm_retrieval
        else:
            raise ValueError(f"Extraction method {method} not supported.")

        # Get total file count by counting all .txt files in the parsed directory
        total_file_count = os.popen(
            f"find {self.config.PARSED_RESULT_DIR} -name '*.txt' | wc -l"
        ).read()
        progress_bar = tqdm(total=total_file_count, desc="Extracting Information")
        for root, dirs, files in sorted(os.walk(self.config.PARSED_RESULT_DIR)):
            for file in files:
                if file.endswith(".txt"):
                    # Check if file is already extracted in extraction_minmod directory
                    if not overwrite:
                        if os.path.exists(
                            f"{self.config.minmod_extraction_dir(method)}/{os.path.splitext(file)[0]}.json"
                        ):
                            logger.info(
                                f"Extraction result for {file} already exists. Skipping."
                            )
                            progress_bar.update(1)
                            continue

                    doc_path = os.path.join(root, file)
                    logger.info(f"Extracting information from {doc_path}")
                    result = extraction_method(doc_path, MineralSite)
                    logger.info(f"Extraction result: {result}")
                    progress_bar.update(1)

        progress_bar.close()


if __name__ == "__main__":
    extractor = MinModExtractor(MODEL_NAME=Config.MODEL_NAME)
    # extractor.bulk_extract("data/asset/parsed_result", ExtractionMethod.BASELINE)

    # Extract individual document
    result = extractor.extract_llm_retrieval(
        "data/asset/parsed_result/Hakkira_Zn_4-2011/Hakkira_Zn_4-2011.txt",
        MineralSite,
    )
