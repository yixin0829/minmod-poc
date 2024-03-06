import os
import re
from typing import Union

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

import prompts as prompts
from config import Config, ExtractionMethod
from schema import (
    BasicInfo,
    DepositTypeCandidate,
    LocationInfo,
    MineralInventory,
    MineralSite,
)

load_dotenv()


class MinModExtractor(object):
    def __init__(self, MODEL: str) -> None:
        self.config = Config()
        self.llm = ChatOpenAI(model=MODEL, temperature=0, max_tokens=2048)

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
        # Check if the file directory exists, if not, create it.
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(file_path, "w") as f:
            f.write(model.json(indent=4))

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
            messages=[
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

        logger.info("Writing the extraction result to a JSON file")
        self._write_model_as_json(
            result,
            f"{self.config.minmod_extraction_dir(ExtractionMethod.BASELINE)}/{file_name}.json",
        )
        logger.info(
            f"Saved at {self.config.minmod_extraction_dir(ExtractionMethod.BASELINE)}/{file_name}.json"
        )

        return result

    def _llm_retrieval_helper(
        self, query: str, doc: str, mineral_site_name: str = None
    ) -> str:
        """
        Use LLM to retrieve information from the document and return a string containing the retrieved information enclosed in <retrieved></retrieved> XML tags.
        """
        parser = StrOutputParser()

        # Construct the prompt for LLM retrieval
        retrieval_prompt = PromptTemplate.from_template(prompts.retrieval_template)
        query_prompt = PromptTemplate.from_template(query)
        doc_prompt = PromptTemplate.from_template(doc)
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=retrieval_prompt,
            pipeline_prompts=[("query", query_prompt), ("doc", doc_prompt)],
        )
        logger.info(
            f"Retrieval pipeline prompt input: {pipeline_prompt.input_variables}"
        )

        chain = pipeline_prompt | self.llm | parser

        retrieval_response = chain.invoke(
            {
                "query": prompts.basic_info_query,
                "doc": doc,
                "mineral_site_name": mineral_site_name,
            }
        )

        # Extract <retrieved> XML tags from the response
        # Regular expression to match the entire segment including <tag> and </tag>
        pattern = r"<retrieved>((.|\n)*?)</retrieved>"

        # Search for the pattern in the string
        match = re.search(pattern, retrieval_response)

        # Extract and print the content including the tags if a match is found
        if match:
            retrieved_info = match.group(1)
            logger.info(f"Retrieved basic information: {retrieved_info}")
        else:
            logger.info("No relevant information retrieved. Set default value.")
            retrieved_info = "I can’t find any relevant information"

        return retrieved_info

    def _llm_extraction_helper(
        self,
        query: str,
        doc: str,
        retrieved_info: str,
        output_schema: BaseModel,
        mineral_site_name: str,
    ) -> Union[BasicInfo, LocationInfo, MineralInventory, DepositTypeCandidate]:

        # Construct the prompt for LLM extraction
        query_prompt = PromptTemplate.from_template(query)
        extraction_prompt = PromptTemplate.from_template(prompts.extraction_template)
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=extraction_prompt,
            pipeline_prompts=[
                ("query", query_prompt),
            ],
        )
        logger.info(
            f"Extraction pipeline prompt input: {pipeline_prompt.input_variables}"
        )

        fix_parser = OutputFixingParser.from_llm(
            PydanticOutputParser(pydantic_object=output_schema), self.llm
        )

        runnable = create_structured_output_runnable(
            output_schema=output_schema,
            llm=self.llm,
            mode="openai-json",
            prompt=pipeline_prompt,
            enforce_function_usage=False,
        )

        runnable = runnable | fix_parser

        result = runnable.invoke(
            {
                "query": query,
                "doc": doc,
                "retrieved_info": retrieved_info,
                "output_schema": output_schema.schema_json(),
                "mineral_site_name": mineral_site_name,
            }
        )

        return result

    def extract_llm_retrieval(
        self, doc_path: str, output_schema: BaseModel
    ) -> MineralSite:
        file_name, doc = self._load_doc(doc_path)

        extraction_result = []
        query_schema_pairs = [
            (prompts.basic_info_query, BasicInfo),
            (prompts.location_info_query, LocationInfo),
            (prompts.mineral_inventory_query, MineralInventory),
            (prompts.deposit_query, DepositTypeCandidate),
        ]

        mineral_site_name = None
        for query, output_schema in query_schema_pairs:
            logger.info(f"Extracting information for query: {query[:60]} ...")

            # Retrieve relevant information
            result_retrieval = self._llm_retrieval_helper(query, doc, mineral_site_name)

            # Extract information
            result_extraction = self._llm_extraction_helper(
                query, doc, result_retrieval, output_schema, mineral_site_name
            )

            # Populate the mineral site name after the basic info extraction
            if type(result_extraction) == BasicInfo:
                logger.info(f"Mineral site name: {result_extraction.name}")
                mineral_site_name = f"{result_extraction.name} Mine"

            extraction_result.append(result_extraction)

        # Construct the MineralSite object
        result = MineralSite(
            basic_info=extraction_result[0],
            location_info=extraction_result[1],
            mineral_inventory=extraction_result[2],
            deposit_type_candidate=extraction_result[3],
        )

        self._write_model_as_json(
            result,
            f"{Config.minmod_extraction_dir(ExtractionMethod.LLM_RETRIEVAL)}/{file_name}.json",
        )

        return result

    def bulk_extract(self, parsed_dir: str, method: ExtractionMethod):
        """Call different extraction methods on all documents in the parsed directory."""
        if method == ExtractionMethod.BASELINE:
            extraction_method = self.extract_baseline
        elif method == ExtractionMethod.LLM_RETRIEVAL:
            extraction_method = self.extract_llm_retrieval
        else:
            raise ValueError(f"Extraction method {method} not supported.")

        # Get total file count by counting all .txt files in the parsed directory
        total_file_count = os.popen(f"find {parsed_dir} -name '*.txt' | wc -l").read()
        progress_bar = tqdm(total=total_file_count, desc="Extracting Information")
        for root, dirs, files in sorted(os.walk(parsed_dir)):
            for file in files:
                if file.endswith(".txt"):
                    doc_path = os.path.join(root, file)
                    logger.info(f"Extracting information from {doc_path}")
                    result = extraction_method(doc_path, MineralSite)
                    logger.info(f"Extraction result: {result}")
                    progress_bar.update(1)
        progress_bar.close()


if __name__ == "__main__":
    extractor = MinModExtractor(MODEL=Config.MODEL)
    # extractor.bulk_extract("data/asset/parsed_result", ExtractionMethod.BASELINE)

    # Extract individual document
    # result = extractor.extract_baseline(
    #     "data/asset/parsed_result/Mehdiabad_Zn_3-2005/Mehdiabad_Zn_3-2005.txt",
    #     MineralSite,
    # )
