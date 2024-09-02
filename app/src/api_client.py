import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
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


    def complete(self, messages, model, temperature, max_tokens, top_p):
        response = self.client.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.choices[0].message.content

    def get_response(self, message):
        return self.complete(
            messages=[UserMessage(content=message)],
            model="Mistral-large-2407",
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
        )

if __name__ == "__main__":
    client = AzureInferenceClient()
    response = client.complete(
        messages=[
            UserMessage(
                content="In what follows, you will ll assume the role of an expert in supply chain management, particularly of an expert trained in operations research applied to supply chains.  You are carrying out a literature review about vaccine supply chains in which you aim to identify some important aspects to be considered in the efficient and robust design of this special distribution network.  You'll be revising a large list of papers with details and you are going to identify in each document, and only in the enlisted documents, the main characteristics of the vaccine supply chain being considered. Specifically, you'll be required to classify the author's work according to the problem being addressed within at most two, preferably one, of the following categories: 1.  Supply Chain Design2. Distribution-allocation-location3. Allocation-location4.  Inventory and Production.  To identify which one is the most likely being addressed, you must focus your attention in the methodological section of each document. Detail the dominant theoretical frameworks exposed in Vaccine supply decisions and government interventions for recurring epidemics by Pan et al."
                )],
        model="Mistral-large-2407",
        temperature=0.7,
        max_tokens=4096,
        top_p=1
    )
    print(response)