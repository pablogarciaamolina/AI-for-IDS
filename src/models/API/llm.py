import time
from abc import abstractmethod
from typing import Any, Optional
import logging

import backoff
import google.api_core
import google.api_core.exceptions
import google.genai.errors
import google.genai.tunings
import google.genai.types
import pandas as pd
import google.genai
import mistralai
import google

from src.models._base import BaseModel
from src.data.utils import jsonize_rows
from src.models.config import MISTRAL_API_KEY, GEMINI_API_KEY, BACKOFF_MAX_TRIES, BACKOFF_FACTOR, GEMINI_GENERATION_CONFIG, MISTRAL_GENERATION_CONFIG
from src.models.utils import log_backoff

class LLModel(BaseModel):
    
    client: Any

    def __init__(self, model: str):
        
        super().__init__(model)
        self.model = model

    @abstractmethod
    def ask(self, instructions: str, context: Optional[str] = None) -> dict[str, str]:
        """
        Interface for chat completion task

        Args:
            instructions: String containing the questions for the model
            context: Optional string containing behavioural context for the model, in order to indicate which role or position to adopt

        Returns:
            :A dictionary containing the reply from the model as well as the exit reason of the model (control purposes). Such like:
                {
                    "answer": [<str contining the answer from the model>, ...(as many as specified in the configuration)],
                    "finish_reason": [<exit reason in str format>, ... (as many as specified in the configuration)]
                }
        """

        pass

    def predict(self,
        icl_data: pd.DataFrame,
        test_data: pd.DataFrame,
        task: str = "Classify data",
        class_column: str = "Label",
        **generation_config
    ) -> list[str]:
        """
        Classifies test input data using an LLM, ensuring the output is one of the possible labels.

        Args:
            icl_data: A pandas DataFrame containing the ICL examples.
            test_data: A pandas DataFrame containing the test data to be classified.
            task: A sentence that further explains the classification task.
            class_column: The name of the column containing the labels in `icl_data`.
            **generation_config: Further configuration for model generation.

        Returns:
            A list containing the lists of predicted labels for each test input. As many predictions (candidates) for each input as specified in the gneration configuration.
        """
        
        possible_labels = set(icl_data[class_column].unique())
        
        icl_inputs = jsonize_rows(icl_data.drop(columns=[class_column]))
        context = (
            f"You are performing the following classification task: {task}\n"
            "Here are some examples:\n"
        )
        for i, o in zip(icl_inputs, icl_data[class_column]):
            context += str({"Input": i, "Output": o}) + "\n"
        context += "\n Where Output is the classification label for each data entry.\n"
        
        pre_instruction = "Classify the following input and provide the corresponding Output:\n"
        last_instruction = (
            "Ensure you provide only the raw Output and nothing else."
        )
        
        results = []
        logging.info('Starting predictions...asking the model...')        
        test_inputs = jsonize_rows(test_data)
        for i, test_input in enumerate(test_inputs):
            logging.info(f'Prediction nº{i+1}...')
            input_instruction = str({"Input": test_input})
            instructions = pre_instruction + input_instruction + "\n" + last_instruction
            context = f"Request ID: {time.time()}\n" + context            
            response = self.ask(instructions=instructions, context=context, **generation_config)

            model_output = []
            for candidate in response["answer"]:
                response_text = candidate.strip()
            
                output = None
                for label in possible_labels:
                    if label in response_text:
                        output = label
                        break
                
                if output is None:
                    logging.warning(f'No valid label found in response: {response_text}')
                    output = None

                model_output.append(output)
            
            logging.info(f'Response: {model_output}')
            results.append(model_output)
        
        return results

class Mistral(LLModel):
    """
    Mistral assitant API following https://docs.mistral.ai/api/
    """

    client: mistralai.Mistral

    def __init__(self, model: str = "mistral-large-latest"):

        super().__init__(model)
        self.client = mistralai.Mistral(api_key=MISTRAL_API_KEY)

    @backoff.on_exception(
        backoff.expo,
        mistralai.models.SDKError,
        max_tries=BACKOFF_MAX_TRIES,
        factor=BACKOFF_FACTOR,
        jitter=backoff.full_jitter,
        giveup=lambda e: "429" not in str(e),
        on_backoff=log_backoff
    )
    def ask(self, instructions: str, context: Optional[str] = None, **generation_config) -> dict[str, list]:

        default_config = MISTRAL_GENERATION_CONFIG.copy()
        default_config.update(generation_config)

        messages: list[dict] = [{
            "role": "system",
            "content": context
        }] if context else []
        messages.append({
            "role": "user",
            "content": instructions
        })

        got_response = False
        while not got_response:

            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                **default_config
            )

            try:
                response_dict = {
                    "answer": [c.message.content for c in response.choices],
                        "finish_reason": [c.finish_reason for c in response.choices]
                }
                got_response = True
            
            except Exception as e:

                logging.warning(f'No response obtained: {e}. Trying again')
                got_response = False

        return response_dict
    
class Gemini(LLModel):

    client: google.genai.Client

    def __init__(self, model: str = "gemini-2.0-pro-exp-02-05"):
        
        super().__init__(model)
        self.client = google.genai.Client(api_key=GEMINI_API_KEY)

    @backoff.on_exception(
        backoff.expo,
        exception=(google.genai.errors.ClientError, google.genai.errors.ServerError),
        max_tries=BACKOFF_MAX_TRIES,
        factor=BACKOFF_FACTOR,
        jitter=backoff.full_jitter,
        giveup=lambda e: not ("429" in str(e) or "503" in str(e)),
        on_backoff=log_backoff
    )
    def ask(self, instructions: str, context: Optional[str] = None, **generation_config) -> dict[str, list]:


        default_config = GEMINI_GENERATION_CONFIG.copy()
        default_config.update(generation_config)

        config = google.genai.types.GenerateContentConfig(
            **default_config
        )
        got_response = False
        while not got_response:

            context = context if context else ""
            response = self.client.models.generate_content(
                model=self.model, contents=context + "\n" + instructions,
                config=config
            )

            try:
                response_dict = {
                    "answer": [c.content.parts[0].text for c in response.candidates],
                    "finish_reason": [c.finish_reason.name for c in response.candidates]
                }
                got_response = True
            
            except Exception as e:

                logging.warning(f'No response obtained: {e}. Trying again')
                got_response = False

        return response_dict