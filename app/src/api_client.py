import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage, AssistantMessage, SystemMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()
class AzureInferenceClient:
    def __init__(self):
        self.client = ChatCompletionsClient(
            endpoint=os.getenv("AZURE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
        )
        print("AzureInferenceClient initialized")
        print(os.getenv("AZURE_ENDPOINT"))
        print(os.getenv("AZURE_KEY"))

        self.message_chain = []


    def complete(self, messages, model, temperature, max_tokens, top_p):
        response = self.client.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        self.message_chain.append(response.choices[0].message)
        return response.choices[0].message.content

    def get_response(self, message):
        return self.complete(
            messages=[UserMessage(content=message)],
            model="Mistral-large-2407",
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
        )

    def append_user_message(self, message):
        if not isinstance(message, str):
            self.message_chain.append(message)
        else:
            self.message_chain.append(UserMessage(content=message))

    def append_response(self, response):
        if not isinstance(response, str):
            self.message_chain.append(response)
        else:
            self.message_chain.append(AssistantMessage(content=response))




if __name__ == "__main__":
    client = AzureInferenceClient()

    first_message = UserMessage(
                content="In what follows, you will  assume the role of an expert in supply chain management, particularly of an expert trained in operations research applied to supply chains.  You are carrying out a literature review about vaccine supply chains in which you aim to identify some important aspects to be considered in the efficient and robust design of this special distribution network.  You'll be revising a large list of papers with details and you are going to identify in each document, and only in the enlisted documents, the main characteristics of the vaccine supply chain being considered. Specifically, you'll be required to classify the author's work according to the problem being addressed within at most two, preferably one, of the following categories: 1.  Supply Chain Design2. Distribution-allocation-location3. Allocation-location4.  Inventory and Production.  To identify which one is the most likely being addressed, you must focus your attention in the methodological section of each document. Detail the dominant theoretical frameworks exposed in Vaccine supply decisions and government interventions for recurring epidemics by Pan et al."
                )

    client.append_user_message(first_message)
    print(client.message_chain)
    response = client.complete(
        messages=client.message_chain,
        model="meta-llama-3.1-405b-instruct",
        temperature=0.7,
        max_tokens=4096,
        top_p=1
    )
    print(response)
    client.append_response(response)
