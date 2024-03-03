import os

from dotenv import load_dotenv
from langchain.chains import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

import prompts as prompts
from config import Config, ExtractionMethod
from schema import MineralSite

load_dotenv()


class MinModExtractor(object):
    def __init__(self, MODEL: str) -> None:
        self.llm = ChatOpenAI(model=MODEL, temperature=0, max_tokens=2048)

    def extract_baseline(self, doc_path: str, output_schema: str) -> MineralSite:
        # Load the document
        logger.info(f"Loading document from {doc_path}")
        with open(doc_path, "r") as f:
            doc = f.read()

        # Construct the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.sys_prompt),
                ("human", "{input}"),
            ]
        )

        # Under the hood, the `create_structured_output_runnable` function uses LCEL to create a chain (e.g. prompt | llm | parser)
        logger.info("Creating structured output runnable")
        runnable = create_structured_output_runnable(
            output_schema=MineralSite,
            llm=self.llm,
            mode="openai-json",
            prompt=prompt,
            enforce_function_usage=False,
        )

        logger.info("Invoking the runnable")
        result = runnable.invoke({"input": doc, "output_schema": output_schema})

        return result


if __name__ == "__main__":
    extractor = MinModExtractor(MODEL=Config.MODEL)
    result = extractor.extract_baseline(
        "data/asset/parsed_result/Reocin_Zn_3-2002/Reocin_Zn_3-2002.txt",
        MineralSite.schema_json(),
    )

    # Turn the result into a JSON string and write to Config.MINMOD_EXTRACTION_DIR as a JSON file with the same name as the document
    os.makedirs(Config.minmod_extraction_dir(ExtractionMethod.BASELINE), exist_ok=True)
    result_json = result.json(indent=4)
    with open(
        f"{Config.minmod_extraction_dir(ExtractionMethod.BASELINE)}/Reocin_Zn_3-2002.json",
        "w",
    ) as f:
        f.write(result_json)
