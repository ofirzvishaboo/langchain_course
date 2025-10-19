from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama



load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be US$500 billion.
    """
    summary_template = f"""
    given the information {information} about a person i want you to create:
    1. a short summary
    2. two interesting facts about then
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5")

    # Configure Ollama connection
    ollama_base_url = "http://localhost:11434"
    ollama_model = "gemma3:270m"

    llm = ChatOllama(
        temperature=0,
        model=ollama_model,
        base_url=ollama_base_url
    )

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)
    # print(response.response_metadata.token_usage.total_tokens)


if __name__ == "__main__":
    main()
