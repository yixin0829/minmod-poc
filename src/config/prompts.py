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

BASIC_INFO_QUERY: str = """the mineral site's name"""

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
