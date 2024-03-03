sys_prompt: str = """You extract information in structured JSON formats. The information of interest includes mineral site's name, location information, critical mineral (e.g. zinc) inventory, and deposit type.

Extract a valid JSON blob from the user input that matches the following JSON Schema:

{output_schema}"""

retrieve_basic_info: str = (
    """Please retrieve from the attached "Bongará Zn 3-2019" report, word-for-word, any paragraph or table that is relevant to the mineral site\'s name, location, coordinate reference system (CRS) used, the country and state or province where the mineral site is located. Please enclose the full list of retrieved paragraphs or tables in <retrieved></retrieved> XML tags. If there are no quotes in this document that seem relevant to this question, please say "I can’t find any relevant information"."""
)

retrieve_inventory: str = (
    """Please retrieve from attached "Bongará Zn 3-2019" mineral report, word-for-word, any paragraph or table that is relevant to the mineral site's indicated, inferred resources or reserves including ore tonnage, grade, cutoff grade. Please enclose the full list of retrieved paragraphs or tables in <retrieved></retrieved> XML tags."""
)

retrieve_deposit_type: str = (
    """Please retrieve from attached "Bongará Zn 3-2019" mineral report, word-for-word, any paragraph or table that is relevant to the question "What's BONGARÁ ZINC PROJECT's deposit type?" Please enclose the full list of retrieved paragraphs or tables in <retrieved></retrieved> XML tags."""
)

basic_extraction: str = """I want you to use a document and relevant information from the document to extract the mineral site's name, location (latitude and longitude), coordinate reference system (CRS) used, the country and state or province where the mineral site is located.

Here is the document, in <document></document> XML tags:
<document>
{{DOCUMENT}}
</document>

Here are direct information retrieved (enclosed in <retrieved></retrieved> XML tags) from the document that are most relevant to the mineral site's name, location (latitude and longitude), coordinate reference system (CRS) used, the country and state or province where the mineral site is located:
<retrieved>
{{INSERT_RELEVANT_QUOTES_HERE}}
</retrieved>

Please use these to construct an answer with extracted entities and format the answer as a JSON object. Ensure that your answer is accurate and doesn’t contain any information not directly supported by the document or the quotes.

Format your answer to a JSON object that conforms to the JSON schema defined in <json></json> XML tags:
<json>
{"properties": {"name": {"description": "The name of the mineral site.", "title": "Name", "type": "string"}, "location": {"default": "Unknown", "description": "The coordinates of the mineral site represented as `POINT(<latitude> <longitude>)` or `POINT(<easting>, <northing>)`", "title": "Location", "type": "string"}, "crs": {"default": "Unknown", "description": "The coordinate reference system (CRS) of the location.", "title": "Crs", "type": "string"}, "country": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "Unknown", "description": "The country where the mineral site is located.", "title": "Country"}, "state_or_province": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "Unknown", "description": "The state or province where the mineral site is located.", "title": "State Or Province"}}, "required": ["name"], "title": "BasicInfo", "type": "object"}
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


##################################### New prompt #####################################
question_1: str = (
    """what's the mineral site's name, or if there are many mineral sites mentioned in the document, names?"""
)
question_2: str = (
    """what's {{mineral_site_name}}'s location (latitude and longitude), coordinate reference system (aka coordinate system) used, the country and state/province where the mineral site is located in?"""
)
question_3: str = """What's {{mineral_site_name}}'s resources or reserves?"""
question_4: str = """"What's {{mineral_site_name}}'s deposit type?"""
user_1: str = """Here is a document, enclosed in <document></document> XML tags:

<document>
{{DOCUMENT}}
</document>

Please retrieve, word-for-word, any paragraphs and tables relevant to the question {{QUESTION}}. Please enclose the full list of retrieved paragraphs and tables in <retrieved></retrieved> XML tags. If there are no paragraphs or tables in this document that seem relevant to this question, please say "I can’t find any relevant information"."""

user_2: str = (
    """I want you to use the attached document and relevant retrieved information from the document to extract the mineral site's name, location (latitude and longitude), coordinate reference system (CRS) used, the country and state or province where the mineral site is located. Format your output to a JSON object that conforms to the JSON schema defined in <json></json> XML tags."""
)
