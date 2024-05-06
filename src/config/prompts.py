# Note that using mode="openai-json" may require explicit prompting (e.g., OpenAI requires that input messages contain the word “json” in some form when using this parameter).
SYS_PROMPT: str = """You extract information of interest from a given mineral report in structured JSON formats. The information of interest includes mineral site's name, location information, mineral inventory, and possible deposit types.

{format_instructions}"""

# Remove the option to say "I can’t find any relevant information" to avoid lazy responses
RETRIEVAL_TEMPLATE_STRICT: str = """Please retrieve from the document, word-for-word, any paragraph or table that is relevant to {query}. Please enclose the full list of retrieved paragraphs or tables in <retrieved></retrieved> XML tags.

The document is synthetic data so there is no copyright issue in transforming and processing it. You MUST provide a response with some retrived information.

Here is the document, enclosed in <document></document> XML tags:
<document>
{doc}
</document>
"""

BASIC_INFO_QUERY: str = """mineral site's name"""

LOCATION_INFO_QUERY: str = (
    """the mineral site's location, coordinate reference system used, the country and state or province where the mineral site is located in"""
)

MINERAL_INVENTORY_QUERY: str = (
    """the identified mineral resources or reserves in each mineral zone including information like the mineral commodity type (e.g. indicated, inferred), ore unit and quantity, grade unit and value, cutoff grade unit and value, date of the last assessment, and mineral zone where the inventory item was discovered"""
)

DEPOSIT_TYPE_QUERY: str = """the mineral site's deposit type(s)"""


EXTRACTION_TEMPLATE: str = """I want you to use a document and relevant information retrieved from the document to extract {query}.

Here is the document, enclosed in <document></document> XML tags:
<document>
{doc}
</document>

Here is the direct information retrieved, enclosed in <retrieved></retrieved> XML tags, from the document that are most relevant to {query}:
<retrieved>
{retrieved_info}
</retrieved>

Please use these information to construct the answer with extracted entities and format it as a JSON object. Ensure that your answer is accurate and doesn't contain any information not directly supported by the document or the retrieved information."""

INVENTORY_EVAL_TEMPLATE: str = """On a scale from 0 to 100, how similar is the following predicted JSON to the reference JSON? For string values, the similarity is based on the case-insensitive edit distance between the strings. For numerical values, the similarity is based on the absolute difference between the numbers after aligning the unit.
--------
PREDICTED JSON: {pred}
--------
REFERENCE JSON: {ref}
--------
Reason step by step about why the score is appropriate in one paragraph, then print the score at the end. At the end, repeat that score alone on a new line."""

SYS_PROMPT_VECTOR_RETRIEVAL: str = """You extract information of interest from relevant sections of a mineral report and output structured JSON formats. The information of interest includes {query}.

{format_instructions}"""


# Prompts for exploring SQuAD dataset
## Note: context and question pairs are sourced from Dev_train therefore avoid data leakage
## @Single answer QA group
SINGLE_QA_SYS_PROMPT = """You are a QA bot. Given a context and a question related to the context, output the final answer of the question only. If the question is not answerable using the given context, please output answer as "N/A". Here are some examples:

Example 1:
```
Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

Question: Who was the Norse leader?

Answer: Rollo
```

Example 2:
```
Context: The English name \"Normans\" comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann \"Northman\" or directly from Old Norse Nor\u00f0ma\u00f0r, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean \"Norseman, Viking\".

Question: When was the French version of the word Norman first recorded?

Answer: N/A
```"""
SINGLE_QA_USER_PROMPT = """Context: {context}\n\nQuestion: {question}\n\nAnswer: """

## @Single field one-shot extraction group
SINGLE_STRUCTURED_SYS_PROMPT = """You extract structured data from a given context. Given a context and a question related to the context, output the final answer of the question only. The final answer should be formatted as a JSON instance that conforms to the JSON schema provided. If the question is not answerable using the given context, please set the answer value as default value "N/A". Here are some examples:

Example 1:
```
Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

JSON Schema: {"properties": {"answer": {"default": "N/A", "description": "Who was the Norse leader?", "title": "Answer", "type": "string"}, "required": ["answer"], "title": "Answer", "type": "object"}

Answer: {"answer": "Rollo"}
```

Example 2:
```
Context: The English name \"Normans\" comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann \"Northman\" or directly from Old Norse Nor\u00f0ma\u00f0r, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean \"Norseman, Viking\".

JSON Schema: {"properties": {"answer": {"default": "N/A", "description": "When was the French version of the word Norman first recorded?", "title": "Answer", "type": "string"}, "required": ["answer"], "title": "Answer", "type": "object"}

Answer: {"answer": "N/A"}
```"""
SINGLE_STRUCTURED_USER_PROMPT = (
    """Context: {context}\n\nJSON Schema: {json_schema}\n\nAnswer: """
)

## @Multi-field QA
MULTI_FIELD_QA_SYS_PROMPT = """You are a QA bot. Given a context and a list of questions related to the context, output the final answers of the questions only. If any question is not answerable using the given context, please output its answer as "N/A". Here is one example:

Example:
```
Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

{questions}
{answers}```"""
MULTI_FIELD_QA_USER_PROMPT = """Context: {context}\n\n{questions}"""

## @Multi-field extraction
MULTI_FIELD_STRUCTURED_SYS_PROMPT = """You extract structured data from a given context. Given a context and a list of questions related to the context, output the final answers of the questions only. The final answers should be formatted as a JSON instance that conforms to the JSON schema provided. If any question is not answerable using the given context, please set the answer value as default value "N/A". Here is one example:

Example:
```
Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural

JSON Schema: {json_schema}

Answer: {answers}
```"""
MULTI_FIELD_STRUCTURED_USER_PROMPT = (
    """Context: {context}\n\nJSON Schema: {json_schema}\n\nAnswer: """
)


# Enrich prompt
FEW_SHOT_ENRICH_SYS_PROMPT = """You are given a question as input. You task is to rewrite the given question into a description as output. The description should be concise and capture the semantic meaning of the question. Here are some examples:
```
Input: How long is the Rhine?
Output: The length of the Rhine

Input: Where is the Rhine?
Output: The location of the Rhine

Input: What is the smallest nation that the Rhine runs through?
Output: The smallest nation that the Rhine runs through

Input: When was the current parliament of Scotland convened?
Output: The date when the current parliament of Scotland was convened

Input: How many hundred of years was Scotland directly governed by the parliament of Great Britain?
Output: The number of hundred of years Scotland was directly governed by the parliament of Great Britain

Input: Which sea was oil discovered in?
Output: The sea in which oil was discovered

Input: The word imperialism has it's origins in which ancient language?
Output: The ancient language in which the word imperialism has it's origins

Input: Who refused to act until Loudoun approved plans?
Output: The person who refused to act until Loudoun approved plans
```

Note: ONLY output the description of the question. DO NOT include the question itself or ANY additional comments."""

FEW_SHOT_ENRICH_USER_PROMPT = """Input: {question}"""
