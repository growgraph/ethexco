from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ethexco.util import frame_content, Frame


def render_response(text: str, llm, onto_str: str | None = None):
    frames_str = "\n".join({f" - **{k}**:{v}" for k, v in frame_content.items()})
    fields = ", ".join([f"`{k}`" for k in frame_content.keys()])

    prompt = f"""
Please process a text - a fragment from a philosophical book.
There are two independent tasks: task A and task B.
Task A. We are creating question / answer pairs for LLM training. Analyze the provided text from the perspective of logical reasoning and provide answers to  the following elements:
        
{frames_str}
        
For Task A follow the instructions:
    
 - each answer must be placed in a block marked correspondingly, eg ```{Frame.CONTEXT} ...``` or ```{Frame.METHOD} ...```.
 - avoid at all cost referencing the text or the author (e.g. "According to the author..."). Use first-person grammatic forms, as if you were the author of the text, when generating fields {Frame.QUESTION_ANSWER_PAIRS}, {Frame.CONTEXT}, {Frame.ASSUMPTIONS}, {Frame.METHOD}, {Frame.THESIS}.
 - all (!!!) {fields} must be present in your answer.

Task B. Generate semantic triples in turtle (ttl) format from the text below.
         
For Task B follow the instructions:
 
 - mark extracted semantic triples as ```ttl ```.
 - use commonly known ontologies (RDFS, OWL, schema etc) to place encountered abstract entities/properties and facts within a broader ontology.
 - entities representing facts must use the namespace `@prefix cd: <https://growgraph.dev/current#> .` 
 - all entities from `cd:` namespace must IMPERATIVELY linked to entities from basic ontologies (RDFS, OWL etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person schema:Organization, etc
 - all facts must form a connected graph with respect to namespace `cd`.
 - make semantic representation as atomic as possible.
 - keep in mind Task B is independent from Task A.

Below is the input text:
        
```input_text

{{input_text}}
```
    """

    parser = StrOutputParser()

    prompt = PromptTemplate(template=prompt, input_variables=["input_text"])

    chain = prompt | llm | parser

    response = chain.invoke({"input_text": text})
    return response
