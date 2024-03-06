# Note that using mode="openai-json" may require explicit prompting (e.g., OpenAI requires that input messages contain the word “json” in some form when using this parameter).
sys_prompt: str = """You extract information of interest from user input in structured JSON formats. The information of interest includes mineral site's name, location information, critical mineral (e.g. zinc) inventory, and possible deposit types.

{format_instructions}"""

retrieval_template: str = """Please retrieve from the document, word-for-word, any paragraph or table that is likely relevant to the question "{query}" Please enclose the full list of retrieved paragraphs or tables in <retrieved></retrieved> XML tags. If there are no paragraphs or tables in this document that seem relevant to this question, please say "I can’t find any relevant information".

Here is the document, enclosed in <document></document> XML tags:
<document>
{doc}
</document>
"""


basic_info_query: str = """what's the mineral site's name this document is about?"""

location_info_query: str = (
    """what's {mineral_site_name}'s location (latitude and longitude), coordinate reference system (aka coordinate system), the country and state/province where {mineral_site_name} is located in?"""
)

mineral_inventory_query: str = (
    """What's {mineral_site_name}'s resources or reserves? What are the mineral commodities, their categories (e.g. indicated, inferred, measured, probable, proven), ore tonnage, grade, cutoff grade, date reported, and zone?"""
)

deposit_query: str = """What's {mineral_site_name}'s deposit type?"""


extraction_template: str = """I want you to use a document and relevant information retrieved from the document to answer the question "{query}"

Here is the document, in <document></document> XML tags:
<document>
{doc}
</document>

Here are direct information retrieved, enclosed in <retrieved></retrieved> XML tags, from the document that are most relevant to the question "{query}":
<retrieved>
{retrieved_info}
</retrieved>

Please use these to construct your answer and format the answer as a JSON object. Ensure that your answer is accurate and doesn’t contain any information not directly supported by the document or the retrieved information.

Format your answer to a JSON object that conforms to the JSON schema defined in <json></json> XML tags:
<json>
{output_schema}
</json>
"""

inventory_extraction: str = """I want you to use a document and relevant retrieved information from the document to extract the mineral site's inferred/indicated/measured mineral resources, or probable/proven mineral reserves. For each mineral resource or mineral reserve, include information like mineral commodity type, ore tonnage, grade, cutoff grade, date, and zone.

Format your output to a JSON object that conforms to the JSON schema defined in <json></json> XML tags:
<json>
{"$defs": {"Commodity": {"const": "Zinc", "title": "Commodity", "type": "string"}, "GradeUnits": {"enum": ["percent", "grams per tonne", "copper equivalence percent", "lead equivalence percent", "US dollar per tonne", "zinc equivalence percent"], "title": "GradeUnits", "type": "string"}, "MineralCategory": {"enum": ["Estimated", "Inferred", "Indicated", "Measured", "Probable", "Proven"], "title": "MineralCategory", "type": "string"}, "WeightUnits": {"enum": ["tonnes", "million tonnes", "kilograms"], "title": "WeightUnits", "type": "string"}}, "properties": {"commodity": {"allOf": [{"$ref": "#/$defs/Commodity"}], "description": "The type of critical mineral."}, "category": {"anyOf": [{"$ref": "#/$defs/MineralCategory"}, {"type": "null"}], "default": "Unknown", "description": "The category of the mineral."}, "ore_unit": {"anyOf": [{"$ref": "#/$defs/WeightUnits"}, {"type": "null"}], "default": "Unknown", "description": "The unit of the ore."}, "ore_value": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": -1, "description": "The value of the ore in the unit of ore_unit", "title": "Ore Value"}, "grade_unit": {"anyOf": [{"$ref": "#/$defs/GradeUnits"}, {"type": "null"}], "default": "Unknown", "description": "The unit of the grade."}, "grade_value": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": -1, "description": "The value of the grade in the unit of grade_unit", "title": "Grade Value"}, "cutoff_grade_unit": {"anyOf": [{"$ref": "#/$defs/GradeUnits"}, {"type": "null"}], "default": "Unknown", "description": "The unit of the cutoff grade. Example: percent (%)"}, "cutoff_grade_value": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": -1, "description": "The value of the cutoff grade in the unit of cutoff_grade_unit", "title": "Cutoff Grade Value"}, "date": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "Unknown", "description": "The date of the mineral inventory in the 'dd-mm-YYYY' format.", "title": "Date"}, "zone": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "Unknown", "description": "The mineral zone where the mineral resources or reserves are located", "title": "Zone"}}, "required": ["commodity"], "title": "MineralInventory", "type": "object"}
</json>

Here is the document, in <document></document> XML tags:
<document>
{{DOCUMENT}}
</document>

Here are direct information retrieved (enclosed in <retrieved></retrieved> XML tags) from the document that are most relevant to the mineral site's inferred/indicated/measured mineral resources, or probable/proven mineral reserves:
<retrieved>
{{INSERT_RELEVANT_QUOTES_HERE}}
</retrieved>

Please use these to construct an answer with extracted entities and format it as a JSON object. Ensure that your answer is accurate and doesn’t contain any information not directly supported by the document or the quotes.
"""


deposit_type_extraction: str = """I want you to use a document and relevant retrieved information from the document to extract the mineral site's deposit type.

Format your output to a JSON object that conforms to the JSON schema defined in <json></json> XML tags:
<json>
{"$defs": {"DepositType": {"enum": ["Supergene zinc", "Siliciclastic-mafic zinc-lead", "MVT zinc-lead", "Irish-type sediment- hosted zinc- lead"], "title": "DepositType", "type": "string"}}, "properties": {"deposit_type": {"allOf": [{"$ref": "#/$defs/DepositType"}], "default": "Unknown", "description": "The type of mineral deposit."}}, "title": "DepositTypeModel", "type": "object"}
</json>

Here is the document, in <document></document> XML tags:
<document>
{{DOCUMENT}}
</document>

Here are direct information retrieved (enclosed in <retrieved></retrieved> XML tags) from the document that are most relevant to the mineral site's deposit type:
<retrieved>
{{INSERT_RELEVANT_QUOTES_HERE}}
</retrieved>

Please use these to construct an answer with extracted entities and format it as a JSON object. Ensure that your answer is accurate and doesn’t contain any information not directly supported by the document or the quotes.
"""
