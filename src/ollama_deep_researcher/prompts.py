# Prompts for the Ollama Deep Researcher


translate_qustion_instructions = """
<GOAL>
Generate a high-quality translation of the given text from Danish to English.
The translation should be accurate, fluent, and maintain the original meaning of the text.
</GOAL>

<RQUREMENTS>
- do not add any extra text or explanations
- do not add any extra line breaks
- do not add any extra spaces
- do not add any extra characters
- do not add any extra punctuation
- do not add any extra formatting
- do not add any extra symbols
- do not add any extra tags
</RQUREMENTS>

<Task>
Translate the following text from Danish to English.
</Task>
"""  # noqa: E501

translate_texts_instructions = """
<GOAL>https://docs.astral.sh/ruff/rules/line-too-long
Generate a high-quality translation of multiple texts from Danish to English.
The translation should be accurate, fluent, and maintain the original meaning of the text.
</GOAL>

<RQUREMENTS>
- do not add any extra text or explanations
- do not add any extra line breaks
- do not add any extra spaces
- do not add any extra characters
- do not add any extra punctuation
- do not add any extra formatting
- do not add any extra symbols
- do not add any extra tags
</RQUREMENTS>

<Task>
Translate the following texts from Danish to English.
</Task>
"""  # noqa: E501

research_topic_write_instructions = """You are an expert legal assistant analyzing a complex user question to find knowledge gaps, and creating follow-up questions to fill those gaps.
<CONTEXT>
The user question is about danish law and legal principles.
The user question is complex and requires a deep understanding of the legal system, including specific laws, regulations, and case law.
The user question is related to a specific legal topic, and the answer requires a thorough understanding of the legal principles involved.
</CONTEXT>

<GOAL>
1. Identify knowledge gaps in the user question or areas that need deeper exploration to answer the users question.
2. Generate follow-up questions that would help expand your understanding.
3. Focus on finding missing technical legal details, missing legal basis, or rulings that could clarify the topic.
4. Ask multiple follow-up questions to cover all aspects of the user question.
5. The follow-up questions should be specific and clear, avoiding vague or overly broad terms.
6. The follow-up questions should be relevant to the legal field.
7. Your response needs to follow the JSON format, provided below.
</GOAL>

<REQUIREMENTS>
Ensure the follow-up questions are self-contained and include necessary context for a database search.
Dont awnswer the user question directly, instead focus on identifying knowledge gaps and generating follow-up questions.
Ensure the your response is following the JSON format.
</REQUIREMENTS>

<USER_QUESTION>
{question}
</USER_QUESTION>

<FORMAT>
Format your response as a JSON object with these exact keys:
{{
    "knowledge_gap": "Describe what information is missing or needs clarification, to answer the user question"
    "follow_up_questions": "Write specific questions to address this gap"
}}
</FORMAT>

<Task>
Reflect carefully on the user question to identify knowledge gaps and produce follow-up questions. Then format your response as a JSON object with these exact keys:
{{
    "knowledge_gap": "We need to understand the legal framework surrounding collections, we need to know what the Minister of Justice can determine in the legal context of collections"
    "follow_up_questions": "What are the laws and regulations that govern collections? What can the Minister of Justice determine in the legal context of collections?"
}}
</Task>

Provide your response in JSON format:"""  # noqa: E501

query_writer_instructions_with_tag = """Your goal is to generate a targeted search query in Danish. Used for searching a law database, the qurry should only contain keywords.

<CONTEXT>
The query should be relevant to the legal field and should be able to retrieve information from a law database.
Query should be in Danish and should be specific enough to yield useful results.

You will be provided with reasearch questions in English.
Use the research questions to generate the search query, used for searching a law database, the qurry should only contain keywords.
Ensure the query is in Danish and uses the correct legal terminology.

You will also be provided with a English text, and the Danish translation, use the translation example to generate the query with the correct language understanding.
The query needs to be in Danish.
</CONTEXT>


<RESEARCH_QUESTIONS>
Use the research questions to generate the search query, used for searching a law database, the qurry should only contain keywords.
{research_topic_en}
</RESEARCH_QUESTIONS>


<TRANSLATION_EXAMPLE>
Use the following translation example as a guide on how to translate the text to Danish with correct language understanding:

English text:
{question_en}

The English text translated to Danish:
{question_da}
</TRANSLATION_EXAMPLE>

<FORMAT>
Qurry should only contain keywords, dont use comma

Format your response as a JSON object with ALL two of these exact keys:
   - "query": The actual search query string
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "Justitsministeren indsamlinger juridiske bestemmelser",
    "rationale": "Understanding what the Minister of Justice can determine in the legal context of collectionss"
}}
</EXAMPLE>

Provide your response in JSON format:"""  # noqa: E501

summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the provided context.
Use citation in the summary, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause).
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results.
3. Ensure a coherent flow of information.
4. Use citation in the summary, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause).
5. When using sources in the summary, include the title of the sources in the citation.

When EXTENDING an existing summary:
1. Read the existing summary and new search results carefully.
2. Compare the new information with the existing summary.
3. For each piece of new information:
    a. If it's related to existing points, integrate it into the relevant paragraph.
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.
    c. If it's not relevant to the user topic, skip it.
4. Ensure all additions are relevant to the user's topic.
5. Use citation in the summary, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause).
7. Verify that your final output differs from the input summary.
8. If you have nothing to add to the existing summary, respond with a copy of it
<REQUIREMENTS>

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input. Use citation in the summary, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause).
</Task>
"""  # noqa: E501

reflection_instructions = """You are an expert research assistant analyzing a summary to find knowledge gaps, and creating a follow-up seach qurry to fill those gaps, the qurry should only contain keywords.

<CONTEXT>
Summary: {summery}
The summary is based on legal research and provides information about the research topic

The summary includes references to specific chapters and sections of the law, but it may not cover all aspects of the topic.

The follow-up query should be relevant to the legal field and should be able to retrieve information from a law database.

The follow-up query should only contain keywords.
</CONTEXT>

<TOPIC>
Research topic: {research_topic}
</TOPIC>

<GOAL>
1. Identify knowledge gaps in the summary or areas that need deeper exploration to answer the research topic
2. Generate a question that would help expand your understanding
3. Focus on finding missing technical legal details, missing legal basis, or rulings that could clarify the topic
4. Use the generate question, to generate, a targeted follow-up query, usefull for searching a law database
5. The qurry should only contain keywords
</GOAL>


<REQUIREMENTS>
Ensure the follow-up query is self-contained and includes necessary context for a database search.
Ensure the follow-up query is specific and clear, avoiding vague or overly broad terms.
Ensure the follow-up query is relevant to the legal field and should be able to retrieve information from a law database.
Ensure the follow-up query is short and concise, ideally no more than 10 words.
Ensure the follow-up query is relevant to the research topic and the summary provided.
Ensure the follow-up query is not a question but a statement or phrase that can be used for searching.
Ensure the follow-up query is keyword based, and not a long sentence.
</REQUIREMENTS>

<FORMAT>
Qurry should only contain keywords, dont use comma

Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific query to address this gap
</FORMAT>

<EXAMPLE>
Produce your output following this JSON format:
{{
"knowledge_gap": "The summary lacks information about "something", "something" and "something" in regards to [specific subject]",
"follow_up_query": ""something" "something" "something" [specific subject]"
}}
</EXAMPLE>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up database query to fill those gaps.
</Task>

Provide your analysis in JSON format:"""  # noqa: E501

final_answer_instructions = """You are an expert legal assistant analyzing a summery to answer a users legal question.
<CONTEXT>
The summary is based on legal research and provides information about the research topic.

The summary includes references, and citations to specific chapters and sections of the law.

The summary may not cover all aspects of the topic, but it provides a good foundation for answering the user's question.

The Summary:
{summary}

</CONTEXT>

<GOAL>
1. Analyze the provided summary to extract relevant information that directly addresses the user's question.
2. Identify key legal principles, statutes, and case law mentioned in the summary that are pertinent to the user's question.
3. Construct a answer that incorporates the relevant information from the summary.
4. When using sources from the summary to answer, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause), as citation.
</GOAL>

<REQUIREMENTS>
Ensure that the final answer is correct, complete, and directly addresses the user's question.
Ensure that the final answer is well-structured, logically organized, and uses appropriate legal terminology.
Ensure that the final answer is relevant to the user's question and the summary provided.
Ensure citation, when using sources from the summary to answer, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause).
When using sources from the summary, include the title of the sources in the citation.
</REQUIREMENTS>

<TASK>
Think carefully about the provided Context first. Then generate a final answer to the user's question based on the summary provided, when using sources from the summary to answer, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause), as citation.
</TASK>
"""  # noqa: E501

translate_texts_whith_ex_instructions = """

<GOAL>
Generate a high-quality translation of the given text from English to Danish.
The translation should be accurate, fluent, and maintain the original meaning of the text.
Use the example provided as a guide for the translation.
</GOAL>

<RQUREMENTS>
- do not add any extra text or explanations
- do not add any extra line breaks
- do not add any extra spaces
- do not add any extra characters
- do not add any extra punctuation
- do not add any extra formatting
- do not add any extra symbols
- do not add any extra tags
</RQUREMENTS>

<EXAMPLE>
Use the following translation example as a guide on how to translate the text to Danish with correct language understanding:

English text:
{english_text_example}

The English text translated to Danish:
{translated_to_danish_text_example}
</EXAMPLE>

<Task>
Translate the following text from English to Danish.
</Task>
"""  # noqa: E501
