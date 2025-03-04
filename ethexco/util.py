import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


import pathlib

from enum import StrEnum


class Frame(StrEnum):
    QUESTION = "Question"
    CONTEXT = "Context"
    ASSUMPTIONS = "Assumptions"
    THESIS = "Thesis"
    METHOD = "Method"


frame_content = {
    Frame.QUESTION: "What question or questions is the author addressing? Is there a clear problem being posed?",
    Frame.CONTEXT: "What is the context of the passage? Is the author referring to a particular place, group of people, time period, or area of activity?",
    Frame.ASSUMPTIONS: "What assumptions does the author make, either explicitly or implicitly?",
    Frame.THESIS: "What is the main statement or conclusion that answers the author’s question(s)?",
    Frame.METHOD: "What logical or philosophical methods does the author use to develop their argument and reach their thesis?",
}


def render_response(text: str, llm, onto_str: str | None = None):
    frames_str = "\n".join({f"- **{k}**:{v}" for k, v in frame_content.items()})

    prompt = f"""
       
        Analyze the provided passage from a philosophical book from the perspective of logical reasoning, identifying the following elements:
        
        {frames_str}  
        
        Return each part in a block marked correspondingly, eg ```{Frame.CONTEXT} ...``` or ```{Frame.METHOD} ...```. Do not use any markup other than that. 
        In case an element can not be identified, skip it.
        Each element should be from first-person point of view, e.g. `{Frame.QUESTION}` should contain direct questions and `{Frame.THESIS}` - first-person statements, e.g. "I assume that ..." or "We will use the following method ..."

        Here is the passsage:
                
        ```
        {{input_text}}
        ```

        """

    parser = StrOutputParser()

    prompt = PromptTemplate(template=prompt, input_variables=["input_text"])

    chain = prompt | llm | parser

    response = chain.invoke({"input_text": text})
    return response


def extract_struct(text, key):
    # Pattern to match text between ```key and ```
    pattern = rf"```{key}(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def crawl_directories(input_path: pathlib.Path, suffixes=(".pdf", ".json")):
    file_paths: list[pathlib.Path] = []

    if not input_path.is_dir():
        print(f"The path {input_path} is not a valid directory.")
        return file_paths

    for file in input_path.rglob("*"):
        if file.is_file() and file.suffix in suffixes:
            file_paths.append(file)
    return file_paths
