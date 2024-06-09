import json

import pydantic

# Note: few shot examples (Norman article) are sourced from dev-v2.0.json. The article is deleted in dev-v2.0.json during official experiments to avoid data leakage.
# @ 1.Batched QA
MULTI_FIELD_QA_SYS_PROMPT = """Your task is to answer a series of questions based on the provided context. If a question cannot be answered using the given context, respond with "N/A"."""
MULTI_FIELD_QA_USER_PROMPT_SHOT1 = """Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

Q[1]: When were the Normans in Normandy?
Q[2]: What is France a region of?
Q[3]: From which countries did the Norse originate?
Q[4]: When did the Frankish identity emerge?"""
MULTI_FIELD_QA_ASSISTANT_PROMPT_SHOT1 = """A[1]: 10th and 11th centuries
A[2]: N/A
A[3]: Denmark, Iceland and Norway
A[4]: N/A"""
MULTI_FIELD_QA_USER_PROMPT_SHOT2 = """Context: To further highlight the difference between a problem and an instance, consider the following instance of the decision version of the traveling salesman problem: Is there a route of at most 2000 kilometres passing through all of Germany's 15 largest cities? The quantitative answer to this particular problem instance is of little use for solving other instances of the problem, such as asking for a round trip through all sites in Milan whose total length is at most 10 km. For this reason, complexity theory addresses computational problems and not particular problem instances.

Q[1]: What is the qualitative answer to this particular problem instance?
Q[2]: By how many kilometers does the traveling salesman problem seek to classify a route between the 15 largest cities in Germany?
Q[3]: What does computational complexity theory most specifically seek to answer?
Q[4]: What does computational simplicity theory most specifically seek to answer?"""
MULTI_FIELD_QA_ASSISTANT_PROMPT_SHOT2 = """A[1]: N/A
A[2]: 2000
A[3]: computational problems
A[4]: N/A"""
MULTI_FIELD_QA_USER_PROMPT = """Context: {context}

{questions}"""


# @ 2.Batched structured extraction
MULTI_FIELD_STRUCTURED_SYS_PROMPT = """Your task is to extract values from the provided context to populate the given JSON schema. Ensure that the output is a valid JSON string conforming to the provided schema. If a value is not present in the context, use "N/A" as the default."""
MULTI_FIELD_STRUCTURED_USER_PROMPT_SHOT1 = """Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

JSON schema: {{"properties": {{"A[1]": {{"default": "N/A", "description": "When were the Normans in Normandy?", "title": "A[1]", "type": "string"}}, "A[2]": {{"default": "N/A", "description": "What is France a region of?", "title": "A[2]", "type": "string"}}, "A[3]": {{"default": "N/A", "description": "From which countries did the Norse originate?", "title": "A[3]", "type": "string"}}, "A[4]": {{"default": "N/A", "description": "When did the Frankish identity emerge?", "title": "A[4]", "type": "string"}}}}, "title": "Answer", "type": "object"}}

Output: """
MULTI_FIELD_STRUCTURED_ASSISTANT_PROMPT_SHOT1 = """{{"A[1]": "10th and 11th centuries", "A[2]": "N/A", "A[3]": "Denmark, Iceland and Norway", "A[4]": "N/A"}}"""
MULTI_FIELD_STRUCTURED_USER_PROMPT_SHOT2 = """Context: To further highlight the difference between a problem and an instance, consider the following instance of the decision version of the traveling salesman problem: Is there a route of at most 2000 kilometres passing through all of Germany's 15 largest cities? The quantitative answer to this particular problem instance is of little use for solving other instances of the problem, such as asking for a round trip through all sites in Milan whose total length is at most 10 km. For this reason, complexity theory addresses computational problems and not particular problem instances.

JSON schema: {{"properties": {{"A[1]": {{"default": "N/A", "description": "What is the qualitative answer to this particular problem instance?", "title": "A[1]", "type": "string"}}, "A[2]": {{"default": "N/A", "description": "By how many kilometers does the traveling salesman problem seek to classify a route between the 15 largest cities in Germany?", "title": "A[2]", "type": "string"}}, "A[3]": {{"default": "N/A", "description": "What does computational complexity theory most specifically seek to answer?", "title": "A[3]", "type": "string"}}, "A[4]": {{"default": "N/A", "description": "What does computational simplicity theory most specifically seek to answer?", "title": "A[4]", "type": "string"}}}}, "title": "Answer", "type": "object"}}

Output: """
MULTI_FIELD_STRUCTURED_ASSISTANT_PROMPT_SHOT2 = """{{"A[1]": "N/A", "A[2]": "2000", "A[3]": "computational problems", "A[4]": "N/A"}}"""
MULTI_FIELD_STRUCTURED_USER_PROMPT = """Context: {context}

JSON schema: {json_schema}

Output: """


# @ 3.Batched QA -> structured extraction
MULTI_FIELD_QA_STRUCTURED_SYS_PROMPT = """Your task is to extract values from the provided context to populate the given JSON schema. Ensure that the output is a valid JSON string conforming to the provided schema. If a value is not present in the context, use "N/A" as the default."""
MULTI_FIELD_QA_STRUCTURED_USER_PROMPT_SHOT1 = """Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
Q[1]: When were the Normans in Normandy?
Q[2]: What is France a region of?
Q[3]: From which countries did the Norse originate?
Q[4]: When did the Frankish identity emerge?
A[1]: 10th and 11th centuries
A[2]: N/A
A[3]: Denmark, Iceland and Norway
A[4]: N/A

JSON schema: {{"properties": {{"A[1]": {{"default": "N/A", "description": "When were the Normans in Normandy?", "title": "A[1]", "type": "string"}}, "A[2]": {{"default": "N/A", "description": "What is France a region of?", "title": "A[2]", "type": "string"}}, "A[3]": {{"default": "N/A", "description": "From which countries did the Norse originate?", "title": "A[3]", "type": "string"}}, "A[4]": {{"default": "N/A", "description": "When did the Frankish identity emerge?", "title": "A[4]", "type": "string"}}}}, "title": "Answer", "type": "object"}}

Output: """
MULTI_FIELD_QA_STRUCTURED_ASSISTANT_PROMPT_SHOT1 = """{{"A[1]": "10th and 11th centuries", "A[2]": "N/A", "A[3]": "Denmark, Iceland and Norway", "A[4]": "N/A"}}"""
MULTI_FIELD_QA_STRUCTURED_USER_PROMPT_SHOT2 = """Context: To further highlight the difference between a problem and an instance, consider the following instance of the decision version of the traveling salesman problem: Is there a route of at most 2000 kilometres passing through all of Germany's 15 largest cities? The quantitative answer to this particular problem instance is of little use for solving other instances of the problem, such as asking for a round trip through all sites in Milan whose total length is at most 10 km. For this reason, complexity theory addresses computational problems and not particular problem instances.
Q[1]: What is the qualitative answer to this particular problem instance?
Q[2]: By how many kilometers does the traveling salesman problem seek to classify a route between the 15 largest cities in Germany?
Q[3]: What does computational complexity theory most specifically seek to answer?
Q[4]: What does computational simplicity theory most specifically seek to answer?
A[1]: N/A
A[2]: 2000
A[3]: computational problems
A[4]: N/A

JSON schema: {{"properties": {{"A[1]": {{"default": "N/A", "description": "What is the qualitative answer to this particular problem instance?", "title": "A[1]", "type": "string"}}, "A[2]": {{"default": "N/A", "description": "By how many kilometers does the traveling salesman problem seek to classify a route between the 15 largest cities in Germany?", "title": "A[2]", "type": "string"}}, "A[3]": {{"default": "N/A", "description": "What does computational complexity theory most specifically seek to answer?", "title": "A[3]", "type": "string"}}, "A[4]": {{"default": "N/A", "description": "What does computational simplicity theory most specifically seek to answer?", "title": "A[4]", "type": "string"}}}}, "title": "Answer", "type": "object"}}

Output: """
MULTI_FIELD_QA_STRUCTURED_ASSISTANT_PROMPT_SHOT2 = """{{"A[1]": "N/A", "A[2]": "2000", "A[3]": "computational problems", "A[4]": "N/A"}}"""
MULTI_FIELD_QA_STRUCTURED_USER_PROMPT = """Context: {context}

JSON schema: {json_schema}

Output: """


class PromptFactory:
    @staticmethod
    def _construct_messages(prompts: list[str]) -> list[dict]:
        # construct the messages, first prompt is always system prompt, followed by user prompt, then system prompt, and so on
        messages = [{"role": "system", "content": prompts[0]}]
        for i, prompt in enumerate(prompts[1:]):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": prompt})

        return messages

    @staticmethod
    def batch_qa_messages(questions: list[str], context: str):
        questions_concat = ""
        for i, q in enumerate(questions):
            questions_concat += f"Q[{i+1}]: {q}\n"

        # remove the last newline character
        questions_concat = questions_concat[:-1]

        user_prompt = MULTI_FIELD_QA_USER_PROMPT.format(
            context=context, questions=questions_concat
        )

        prompts = (
            MULTI_FIELD_QA_SYS_PROMPT,
            MULTI_FIELD_QA_USER_PROMPT_SHOT1,
            MULTI_FIELD_QA_ASSISTANT_PROMPT_SHOT1,
            MULTI_FIELD_QA_USER_PROMPT_SHOT2,
            MULTI_FIELD_QA_ASSISTANT_PROMPT_SHOT2,
            user_prompt,
        )

        messages = PromptFactory._construct_messages(prompts)

        return messages

    @staticmethod
    def batch_structured_messages(questions: list[str], context: str):
        schema_fields = {}
        for i, q in enumerate(questions):
            schema_fields[f"A[{i+1}]"] = (
                str,
                pydantic.Field(default="N/A", description=q),
            )
        AnswerModel = pydantic.create_model("Answer", **schema_fields)
        json_schema_dict = AnswerModel.model_json_schema()
        json_schema_string = json.dumps(json_schema_dict)

        user_prompt = MULTI_FIELD_STRUCTURED_USER_PROMPT.format(
            context=context,
            json_schema=json_schema_string,
        )

        prompts = (
            MULTI_FIELD_STRUCTURED_SYS_PROMPT,
            MULTI_FIELD_STRUCTURED_USER_PROMPT_SHOT1,
            MULTI_FIELD_STRUCTURED_ASSISTANT_PROMPT_SHOT1,
            MULTI_FIELD_STRUCTURED_USER_PROMPT_SHOT2,
            MULTI_FIELD_STRUCTURED_ASSISTANT_PROMPT_SHOT2,
            user_prompt,
        )

        messages = PromptFactory._construct_messages(prompts)

        return messages

    @staticmethod
    def batch_qa_structured_messages(
        questions: list[str], qa_response: str, context: str
    ):
        # construct the new context = context + question & answer pairs
        qa_pairs_concat = ""
        for i, q in enumerate(questions):
            qa_pairs_concat += f"Q[{i+1}]: {q}\n"
        qa_pairs_concat += qa_response
        context = context + "\n" + qa_pairs_concat

        # construct json schema based on the questions
        schema_fields = {}
        for i, q in enumerate(questions):
            schema_fields[f"A[{i+1}]"] = (
                str,
                pydantic.Field(default="N/A", description=q),
            )
        AnswerModel = pydantic.create_model("Answer", **schema_fields)
        json_schema_dict = AnswerModel.model_json_schema()
        json_schema_string = json.dumps(json_schema_dict)

        user_prompt = MULTI_FIELD_QA_STRUCTURED_USER_PROMPT.format(
            context=context, json_schema=json_schema_string
        )

        prompts = (
            MULTI_FIELD_QA_STRUCTURED_SYS_PROMPT,
            MULTI_FIELD_QA_STRUCTURED_USER_PROMPT_SHOT1,
            MULTI_FIELD_QA_STRUCTURED_ASSISTANT_PROMPT_SHOT1,
            MULTI_FIELD_QA_STRUCTURED_USER_PROMPT_SHOT2,
            MULTI_FIELD_QA_STRUCTURED_ASSISTANT_PROMPT_SHOT2,
            user_prompt,
        )

        messages = PromptFactory._construct_messages(prompts)

        return messages
