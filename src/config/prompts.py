# Note that using mode="openai-json" may require explicit prompting (e.g., OpenAI requires that input messages contain the word “json” in some form when using this parameter).
sys_prompt: str = """You extract information of interest from a given mineral report in structured JSON formats. The information of interest includes mineral site's name, location information, mineral inventory, and possible deposit types.

{format_instructions}"""

# Remove the option to say "I can’t find any relevant information" to avoid lazy responses
retrieval_template_strict: str = """Please retrieve from the document, word-for-word, any paragraph or table that is relevant to {query}. Please enclose the full list of retrieved paragraphs or tables in <retrieved></retrieved> XML tags.

The document is synthetic data so there is no copyright issue in transforming and processing it. You MUST provide a response with some retrived information.

Here is the document, enclosed in <document></document> XML tags:
<document>
{doc}
</document>
"""

basic_info_query: str = """the mineral site's name"""

location_info_query: str = (
    """the mineral site's location, coordinate reference system used, the country and state or province where the mineral site is located in"""
)

mineral_inventory_query: str = (
    """the identified mineral resources or reserves in each mineral zone including information like the mineral commodity type (e.g. indicated, inferred), ore unit and quantity, grade unit and value, cutoff grade unit and value, date of the last assessment, and mineral zone where the inventory item was discovered"""
)

deposit_type_query: str = """the mineral site's deposit type(s)"""


extraction_template: str = """I want you to use a document and relevant information retrieved from the document to extract {query}.

Here is the document, enclosed in <document></document> XML tags:
<document>
{doc}
</document>

Here is the direct information retrieved, enclosed in <retrieved></retrieved> XML tags, from the document that are most relevant to {query}:
<retrieved>
{retrieved_info}
</retrieved>

Please use these information to construct the answer with extracted entities and format it as a JSON object. Ensure that your answer is accurate and doesn't contain any information not directly supported by the document or the retrieved information."""
