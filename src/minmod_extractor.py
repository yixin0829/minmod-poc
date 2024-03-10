import datetime
import os
import re

import tqdm
from dotenv import load_dotenv
from langchain.chains import create_structured_output_runnable
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic.v1 import BaseModel
from pydantic.v1.error_wrappers import ValidationError

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


class MinModExtractor(object):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
        )

    def extract_baseline(self, doc: str, output_schema: BaseModel) -> BaseModel:
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
        logger.info("Creating baseline structured extraction chain")
        chain = prompt | llm | parser_fixing

        logger.info("Invoking the chain to extract info from input")
        result = chain.invoke({"input": doc})

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

        logger.info("Invoking the chain to extract info")
        result = chain.invoke(
            {"query": query, "doc": doc, "retrieved_info": retrieved_info}
        )

        return result

    def extract_llm_retrieval(
        self, doc: str, output_schema: MineralSite
    ) -> MineralSite:
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
        except ValidationError as e:
            logger.error(f"Failed to validate the extraction result: {e}")

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
            extraction_func = self.extract_baseline
        elif method == ExtractionMethod.LLM_RETRIEVAL:
            extraction_func = self.extract_llm_retrieval
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


if __name__ == "__main__":
    config = Config()
    extractor = MinModExtractor(config)
    # extractor.bulk_extract(ExtractionMethod.BASELINE, overwrite=Config.MINMOD_BULK_EXTRACTION_OVERWRITE)

    # Extract individual document
    docs_w_ground_truth = [
        "data/asset/parsed_result/Bleiberg_Pb_Zn_5-2017/Bleiberg_Pb_Zn_5-2017.txt",
        "data/asset/parsed_result/Bongará_Zn_3-2019/Bongará_Zn_3-2019.txt",
        "data/asset/parsed_result/Hakkari_Zn_3-2010/Hakkari_Zn_3-2010.txt",
        "data/asset/parsed_result/Hakkari_Zn_7-2013/Hakkari_Zn_7-2013.txt",
        "data/asset/parsed_result/Hakkira_Zn_4-2011/Hakkira_Zn_4-2011.txt",
        "data/asset/parsed_result/Mehdiabad_Zn_3-2005/Mehdiabad_Zn_3-2005.txt",
        "data/asset/parsed_result/Reocin_Zn_3-2002/Reocin_Zn_3-2002.txt",
    ]

    docs_mock = [
        "data/mock/Bongará_Zn_3-2019.txt",
    ]

    file_name, doc = load_doc(docs_mock[0])
    # result = extractor.extract_baseline(doc, MineralSite)
    result = extractor.extract_llm_retrieval(doc, MineralSite)
