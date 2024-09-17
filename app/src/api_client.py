import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage, AssistantMessage, SystemMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()


class AzureInferenceClient:
    def __init__(self,
                 model="Mistral-large-2407",
                 temperature=0.7,
                 max_tokens=4096,
                 top_p=1
                 ):
        self.client = ChatCompletionsClient(
            endpoint=os.getenv("AZURE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
        )
        print("AzureInferenceClient initialized")
        self.message_chain = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def complete(self, messages):
        response = self.client.complete(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p
        )
        self.message_chain.append(response.choices[0].message)
        return response.choices[0].message.content

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

class ArticleSection:
    def __init__(self, title, content):
        self.title = title
        self.content = content

    def __str__(self):
        return f"{self.title}\n{self.content}"

class SystematicReview:
    def __init__(self,
                 subject,
                 article_list,
                 model="Meta-Llama-3-70B-Instruct",
                 temperature=0.7,
                 max_tokens=4096,
                 top_p=1
                 ):

        self._title = None
        self._headings = []
        self._abstract = None
        self._article_sections = []



        self.client = self.set_client()

        print("Systematic  Review client initialized")
        self.subject = subject
        self.article_list = article_list

        self.message_chain = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def set_client(self):
        return ChatCompletionsClient(
            endpoint=os.getenv("AZURE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
        )

    @property
    def article_list(self):
        return self._article_list

    @article_list.setter
    def article_list(self, value):
        self._article_list = value

    @property
    def headings(self):
        return self._headings

    @headings.setter
    def headings(self, value):
        self._headings = value

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def abstract(self):
        return self._abstract

    @abstract.setter
    def abstract(self, value):
        self._abstract = value

    def retrieve_title(self, subject, prompt=None):
        if prompt is None:
            prompt = prompt0 = f"There is an academic puzzle for you. I will give you a list of references. In subject {subject}, a survey paper existing with these references. You will give me a guess of the title of the survey paper. Here are the references, each preceded by the word ARTICLE and a number: {str([f"ARTICLE {i + 1}: {ref}" for i, ref in enumerate(self.article_list)])} The output should be in several lines, and the content in the last one is your answer (the title that you guess). Only one guess is required. The title should start with 'Title:'."
        else:
            prompt = prompt.__format__(subject)
        self.append_user_message(prompt)
        response = self.complete(messages=self.message_chain)
        self.append_response(response)
        self._title = response.split("Title:")[-1]

    def retrieve_abstract(self):
        prompt = f"""
        You are an academic paper writing assistant in the subject {self.subject}.
        I am writing a survey paper titled with {self.title}.
        Here is my outline : 

         {[heading for heading in self.headings]}

        Can you write an abstract for me upon the basis of the provided references?
        You may write only 1 paragraph with about 200-500 words, do not include more
        detailed headings, or any lists.
        You should focus on the content of the abstract, which should be limited to the context available in the references, do not repeat the word “abstract” or include any other content. 
        """

        self.append_user_message(prompt)
        response = self.complete(messages=self.message_chain)
        self.append_response(response)
        self._abstract = response


    def complete(self, messages):
        response = self.client.complete(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p
        )
        self.message_chain.append(response.choices[0].message)
        return response.choices[0].message.content

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

    def append_articles(self, article):
        if isinstance(article, list):
            self.article_list.extend(article)
        else:
            self.article_list.append(article)

    def get_headings(self, n_sections=6, prompt=None):
        prompt = f"Can you guess the outline of this paper? Just generate a list of first-level headings. About {n_sections} first-level headings are good! Just output the first-level headings, do not generate any other content. No item number is required, each first-level heading begins with '*'."
        self.append_user_message(prompt)
        response = self.complete(messages=self.message_chain)
        self.append_response(response)
        self.headings = self.get_sections(response)

    def get_sections(self, response, separator="*"):
        response = response.split(separator)
        # remove first element
        response.pop(0)

        return response

    def select_sections_references(self):
        prompt = """You are an academic paper writing assistant in the subject {}.
        I am writing a survey paper titled with {}.
        Here is my abstract : {}
        
        I already gave you a list of references. Can you help me choose from these references
        that might be useful when I write this chapter the section {}?
        The output should be in several lines. References that you think may be useful
        should be on one line each, beginning with “*”.
        Please retain the square bracketed numbers (like [1], [20]) I give for each reference."""
        for heading in self.headings:
            try:
                prompt = prompt.format(self.subject, self.title, self.abstract, heading)
                self.append_user_message(prompt)
                response = self.complete(messages=self.message_chain)
                self._article_sections.append(ArticleSection(heading, response))
                self.message_chain.pop()
            except Exception as e:
                print(f" Error: {e}")
                break

    def write_review(self,  params={}):
        if self.subject is None:
            raise "Cannot write review about an unspecified topic"

        self.retrieve_title(self.subject, params.get("title_prompt", None))
        self.get_headings(params.get("n_sections", 6), params.get("headings_prompt", None))
        self.retrieve_abstract()
        self.select_sections_references()
        self.print_review()

    def print_review(self):

        print(f"Title: {self.title}")
        print(f"Headings: \n{self.headings}")
        print(f"Abstract: {self.abstract}")
        print("Sections:")
        for section in self._article_sections:
            print(section)



if __name__ == "__main__":
    ArticleList = [
        "ALee, B.Y., Assi, T.M., Rookkapan, K., Connor, D.L., Rajgopal, J., Sornsrivichai, V., Brown, S.T., Welling, J.S., Norman, B.A., Chen, S.I., et al., 2011. Replacing the measles ten-dose vaccine presentation with the single-dose presentation in thailand. Vaccine 29, 21, 3811–3817.",
        "Lee, B.Y., Cakouros, B.E., Assi, T.M., Connor, D.L., Welling, J., Kone, S., Djibo, A., Wateska, A.R., Pierre, L., Brown, S.T., 2012. The impact of making vaccines thermostable in niger’s vaccine supply chain. Vaccine 30, 38, 5637–5643.",
        "Sun, H., Toyasaki, F., Falagara Sigala, I., 2023. Incentivizing at-risk production capacity building for covid-19 vaccines. Production and Operations Management 32, 5, 1550–1566.",
        "Kis, Z., Kontoravdi, C., Shattock, R., Shah, N., 2020. Resources, production scales and time required for producing rna vaccines for the global pandemic demand. Vaccines 9, 1, 3. ",
        "Hovav, S., Tsadikovich, D., 2015. A network flow model for inventory management and distribution of influenza vaccines through a healthcare supply chain. Operations Research for Health Care 5, 49–62. ",
        "Demirci, E.Z., Erkip, N.K., 2020. Designing intervention scheme for vaccine market: a bilevel programming approach. Flexible Services and Manufacturing Journal 32, 2, 453–485.",
        "Lin, Q., Zhao, Q., Lev, B., 2020. Cold chain transportation decision in the vaccine supply chain. European Journal of Operational Research 283, 1, 182–195.",
        "Tavana, M., Govindan, K., Nasr, A.K., Heidary, M.S., Mina, H., 2021. A mathematical programming approach for equitable covid-19 vaccine distribution in developing countries. Annals of Operations Research ",
        "Goodarzian, F., Navaei, A., Ehsani, B., Ghasemi, P., Mu˜ nuzuri, J., 2022. Designing an integrated responsive-green-cold vaccine supply chain network using internet-of-things: artificial intelligence-based solutions. Annals of Operations Research ",
        "Dastgoshade, S., Shafiee, M., Klibi, W., Shishebori, D., 2022. Social equity-based distribution networks design for the covid19 vaccine. International Journal of Production Economics 250, 108684. Special Issue celebrating Volume 250 of the International Journal of Production Economics.",
        "Rahman, H.F., Chakrabortty, R.K., Paul, S.K., Elsawah, S., 2023. Optimising vaccines supply chains to mitigate the covid-19 pandemic. International Journal of Systems Science: Operations & Logistics 10, 1, 2122757. ",
        "Mak, H.Y., Dai, T., Tang, C.S., 2022. Managing two-dose covid-19 vaccine rollouts with limited supply: Operations strategies for distributing time-sensitive resources. Production and operations management 31, 12, 4424–4442.",
        "Basciftci, B., Yu, X., Shen, S., 2023. Resource distribution under spatiotemporal uncertainty of disease spread: Stochastic versus robust approaches. Computers & Operations Research 149, 106028.",
        "Mofrad, M.H., Garcia, G.G.P., Maillart, L.M., Norman, B.A., Rajgopal, J., 2016. Customizing immunization clinic operations to minimize open vial waste. Socio-Economic Planning Sciences 54, 1–17. ",
        "Lim, J., Claypool, E., Norman, B.A., Rajgopal, J., 2016. Coverage models to determine outreach vaccination center locations in low and middle income countries. Operations research for health care 9, 40–48.",
        "Kayvanfar, V., Husseini, S.M., Karimi, B., Sajadieh, M.S., 2017. Bi-objective intelligent water drops algorithm to a practical multi-echelon supply chain optimization problem. Journal of Manufacturing Systems 44, 93–114. ",
        "Qi, Y., Liao, K., Liu, T., Zhang, Y., 2022. Originating multiple-objective portfolio selection by counter-covid measures and analytically instigating robust optimization by mean-parameterized nondominated paths. Operations Research Perspectives 9, 100252.",
        "Ng, C., Cheng, T., Tsadikovich, D., Levner, E., Elalouf, A., Hovav, S., 2018. A multi-criterion approach to optimal vaccination planning: Method and solution. Computers & Industrial Engineering 126, 637–649. ",
        "Zhang, C., Li, Y., Cao, J., Wen, X., 2022. On the mass covid-19 vaccination scheduling problem. Computers & Operations Research 141, 105704",
        "Fadaki, M., Abareshi, A., Far, S.M., Lee, P.T.W., 2022. Multi-period vaccine allocation model in a pandemic: A case study of covid-19 in australia. Transportation Research Part E: Logistics and Transportation Review 161, 102689."
    ]
    client = SystematicReview(subject="vaccines supply chain management", article_list=ArticleList)

    client.retrieve_title("vaccines supply chain management")
    client.write_review()

