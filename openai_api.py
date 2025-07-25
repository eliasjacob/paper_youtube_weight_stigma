import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field, validator

ZERO_SHOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um especialista em análise de sentimento com foco em comentários relacionados a peso corporal. "
            "Sua tarefa é classificar comentários como positivos, negativos ou neutros, com atenção especial para detectar gordofobia. "
            "\n\nSiga estas diretrizes:"
            "\n1. Classifique o sentimento geral do comentário como 'positivo', 'negativo' ou 'neutro'."
            "\n2. Identifique o idioma do texto, usando códigos de duas letras (ex: 'pt', 'en', 'es')."
            "\n3. Comentários com discriminação por peso (gordofobia) geralmente devem ser classificados como negativos."
            "\n4. Considere o contexto completo do comentário, não apenas palavras isoladas."
            "\n5. Comentários de apoio, aceitação e respeito ao corpo devem ser considerados positivos."
            "\n6. Comentários informativos sem julgamento de valor são tipicamente neutros."
            "\n\nResponda usando exatamente o formato solicitado, sem adicionar informações extras.",
        ),
        ("human", "{text}"),
    ]
)

def generate_ZERO_SHOT_PROMPTs(input_text: str) -> Dict[str, Any]:
    """Generate few-shot NER prompts using a given example text.

    Args:
        input_text (str): The input text example for generating NER prompts.

    Returns:
        Dict[str, Any]: A dictionary containing the generated NER prompts.
    """
    # Invoke the ZERO_SHOT_PROMPT with the provided input text and formatted examples
    ner_prompts = ZERO_SHOT_PROMPT.invoke(dict(
        text=input_text,
    ))
    
    return ner_prompts


def get_formatted_tool_configuration(
    model_name: str, 
    base_model: BaseModel, 
    model_url: str | None = None
) -> dict:
    """Get the tool configuration dictionary for different LLM providers.

    Args:
        model_name: Name of the language model (e.g., 'gpt-4', 'claude-2').
        base_model: Pydantic model defining the expected output schema.
        model_url: URL for Ollama model endpoint. Required only for Ollama models.

    Returns:
        Dictionary containing tool configuration for the specified model.

    Raises:
        ValueError: If model_url is not provided for Ollama models.

    Example:
        >>> schema = MyOutputSchema()
        >>> config = get_formatted_tool_configuration("gpt-4", schema)
        >>> tools = config['tools']
    """
    if model_name.startswith("gpt"):
        # Configure tools for GPT models
        tool_configuration = ChatOpenAI(model=model_name, timeout=180, temperature=0.0).with_structured_output(base_model, method='json_schema').first.kwargs
    else:
        # Handle Ollama models
        print(f"Handling model {model_name} as Ollama")
        if model_url is None:
            raise ValueError("model_url must be provided for Ollama models")
        tool_configuration = ChatOllama(
            model=model_name, 
            timeout=360, 
            num_ctx=16384,
            num_predict=-1,
            base_url=model_url,
            temperature=0.0).with_structured_output(base_model).first.kwargs

    # Extract the 'tools' key from the configuration
    print(f"Tools: {tool_configuration}")
    tools = tool_configuration['tools']
    return tools

def create_openai_format_request(
    model_name: str, 
    base_model: BaseModel, 
    request_id: str, 
    messages: List[Dict[str, Any]], 
    max_tokens: int = 4096, 
    tool_choice_name: str | None = None
) -> dict:
    """Create a formatted request for OpenAI API in JSONL format.

    Args:
        model_name: Name of the language model to use.
        base_model: Pydantic model defining the expected output schema.
        request_id: Unique identifier for the request.
        messages: List of conversation messages.
        max_tokens: Maximum tokens in response, defaults to 4096.
        tool_choice_name: Name of the tool to use, defaults to first available tool.

    Returns:
        Dictionary containing the formatted request entry.

    Example:
        >>> schema = MyOutputSchema()
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> request = create_openai_format_request("gpt-4", schema, "req1", messages)
    """
    tools = get_formatted_tool_configuration(model_name, base_model)
    if tool_choice_name is None:
        tool_choice_name = tools[0]['function']['name']

    return {
        'custom_id': request_id,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': model_name,
            'messages': messages,
            'max_tokens': max_tokens,
            'tools': tools,
            'parallel_tool_calls': False,
            'tool_choice': {
                'type': 'function',
                'function': {
                    'name': tool_choice_name
                }
            }
        }
    }

def save_jsonl_batches(
    base_file_path: str, 
    entries: List[dict], 
    batch_size: int = 3000
) -> List[str]:
    """Save JSONL entries to files, splitting into batches if needed.

    Args:
        base_file_path: Base path for output files.
        entries: List of entries to write to JSONL files.
        batch_size: Maximum entries per file, defaults to 3000.

    Returns:
        List of paths to created JSONL files.

    Example:
        >>> entries = [{"id": 1}, {"id": 2}]
        >>> files = save_jsonl_batches("output/data", entries)
        >>> print(files)  # ['output/data_0.jsonl']
    """
    output_paths = []

    if len(entries) > batch_size:
        num_files = len(entries) // batch_size
        for i in range(num_files + 1):
            file_path = f'{base_file_path}_{i}.jsonl'
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(entries))
            
            with open(file_path, 'w') as file:
                for entry in entries[start_idx:end_idx]:
                    file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            output_paths.append(file_path)
            print(f'Created file: {file_path}')
    else:
        with open(base_file_path, 'w') as file:
            for entry in entries:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        output_paths.append(base_file_path)
        print(f'Created file: {base_file_path}')
    
    return output_paths

def convert_langchain_to_openai_messages(
    langchain_prompt: ChatPromptTemplate
) -> List[Dict[str, Any]]:
    """Convert LangChain prompt messages to OpenAI API format.

    Args:
        langchain_prompt: LangChain few-shot NER prompt template.

    Returns:
        List of messages formatted for OpenAI API.

    Example:
        >>> template = ChatPromptTemplate.from_messages([...])
        >>> messages = convert_langchain_to_openai_messages(template)
    """
    openai_messages = []
    # Mapping message classes to OpenAI roles
    role_mapping = {
        'SystemMessage': 'system',
        'HumanMessage': 'user',
        'AIMessage': 'assistant',
        'ToolMessage': 'function'
    }

    # Keep track of tool call IDs to function names
    tool_call_id_to_name = {}

    # Iterate over each message in the prepared prompt
    for message in langchain_prompt.messages:
        message_class_name = type(message).__name__
        role = role_mapping.get(message_class_name)
        content = getattr(message, 'content', None)
        
        new_message = {
            'role': role,
            'content': content,
        }

        # Handle tool calls in AI messages
        if role == 'assistant':
            # Check if message has 'tool_calls' attribute
            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                # Assume only one tool_call per message
                tool_call = tool_calls[0]
                tool_call_id = tool_call['id']
                function_name = tool_call['name']
                tool_call_id_to_name[tool_call_id] = function_name
                new_message['function_call'] = {
                    'name': function_name,
                    'arguments': json.dumps(tool_call['args'], ensure_ascii=False)
                }
                # When function_call is present, content should be None
                new_message['content'] = None

        # For 'function' messages (originally 'ToolMessage' instances)
        elif role == 'function':
            # Get tool_call_id from message
            tool_call_id = getattr(message, 'tool_call_id', None)
            function_name = None
            if tool_call_id:
                function_name = tool_call_id_to_name.get(tool_call_id)
            else:
                # If function name is directly available
                function_name = getattr(message, 'name', None)

            if function_name:
                new_message['name'] = function_name
            else:
                # If function name cannot be determined
                new_message['name'] = "UnknownFunction"

        openai_messages.append(new_message)
    
    return openai_messages

class BatchJob(BaseModel):
    id: str
    status: str
    created_at: int
    expires_at: int
    input_file_id: str
    output_file_id: Optional[str] = None
    endpoint: str
    completion_window: str
    metadata: dict

class BatchOutput(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

class OpenAIBatchProcessor:
    def __init__(self):
        self.client = OpenAI()
        self.last_batch = None


    def submit_batch_job(self, input_jsonl_path: Path, batch_name: str, type: str = 'completions') -> BatchJob:
        """Submit a batch job to the OpenAI API and save the batch ID to a file.

        Args:
            input_jsonl_path (Path): The path to the input JSONL file.
            batch_name (str): The name of the batch job.

        Returns:
            BatchJob: The batch job information returned by the OpenAI API.
        """
        if isinstance(input_jsonl_path, str):
            input_jsonl_path = Path(input_jsonl_path)
        
        with open(input_jsonl_path, "rb") as file:
            batch_input_file = self.client.files.create(file=file, purpose="batch")
        
        batch_input_file_id = batch_input_file.id

        if type == 'completions':
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": batch_name
                }
            )
        elif type == 'embeddings':
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": batch_name
                }
            )
        print(f'Successfully submitted batch {batch_name} with id {batch_job.id}')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id_file_path = input_jsonl_path.parent / f"{input_jsonl_path.stem}_{timestamp}.txt"
        
        with open(batch_id_file_path, "w") as batch_id_file:
            batch_id_file.write(str(batch_job))
        
        print(f'Batch info saved to {batch_id_file_path}')
        
        self.last_batch = BatchJob(**batch_job.model_dump())
        return self.last_batch


    def is_batch_complete(self, batch_id: Optional[str] = None) -> bool:
        """Check if a batch job is complete.

        Args:
            batch_id (Optional[str]): The ID of the batch job. If not provided, uses the last batch ID.

        Returns:
            bool: True if the batch job is complete, False otherwise.
        """
        batch_id = self._check_batch_id(batch_id)

        batch_job = self.client.batches.retrieve(batch_id)
        return batch_job.status == "completed"

    def get_batch_info(self, batch_id: Optional[str] = None) -> BatchJob:
        """Retrieve information about a batch job.

        Args:
            batch_id (Optional[str]): The ID of the batch job. If not provided, uses the last batch ID.

        Returns:
            BatchJob: The batch job information.
        """
        batch_id = self._check_batch_id(batch_id)

        batch_job = self.client.batches.retrieve(batch_id)
        return BatchJob(**batch_job.model_dump())

    def get_batch_output(self, batch_id: Optional[str] = None) -> List[BatchOutput]:
        """Retrieve the output of a batch job.

        Args:
            batch_id (Optional[str]): The ID of the batch job. If not provided, uses the last batch ID.

        Returns:
            List[BatchOutput]: The output of the batch job.
        """
        batch_id = self._check_batch_id(batch_id)

        assert self.is_batch_complete(batch_id), f"Batch {batch_id} is not complete"
        file_response = self.client.files.content(self.client.batches.retrieve(batch_id).output_file_id)
        return [line for line in file_response.iter_lines()]
    
    def get_parsed_output(self, base_model: BaseModel, batch_id: Optional[str] = None) -> List[BaseModel]:
        """Retrieve the parsed output of a batch job.

        Args:
            base_model (BaseModel): The base model schema to parse the output.
            batch_id (Optional[str]): The ID of the batch job. If not provided, uses the last batch ID.

        Returns:
            List[BaseModel]: The parsed output of the batch job.
        """
        batch_id = self._check_batch_id(batch_id)

        file_response = self.get_batch_output(batch_id)
        parsed_outputs = []

        for output in file_response:
            json_output = json.loads(output)
            json_output = json.loads(json_output['response']['body']['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            obj = base_model.parse_obj(json_output)
            parsed_outputs.append(obj)

        return parsed_outputs

    def list_pending_batches(self) -> List[BatchJob]:
        """List all pending batch jobs.

        Returns:
            List[BatchJob]: A list of pending batch jobs.
        """
        batches = self.client.batches.list()
        pending_batches = [BatchJob(**batch.model_dump()) for batch in batches if batch.status == "in_progress"]
        return pending_batches

    def list_last_batches(self, limit: int = 5) -> List[BatchJob]:
        """List the last N batch jobs.

        Args:
            limit (int, optional): The number of last batch jobs to list. Defaults to 5.

        Returns:
            List[BatchJob]: A list of the last N batch jobs.
        """
        batches = self.client.batches.list(limit=limit)
        return [BatchJob(**batch.model_dump()) for batch in batches.data]

    def cancel_batch_job(self, batch_id: str) -> BatchJob:
        """Cancel a batch job.

        Args:
            batch_id (str): The ID of the batch job to cancel.

        Returns:
            BatchJob: The batch job information after cancellation.
        """
        batch_job = self.client.batches.cancel(batch_id)
        return BatchJob(**batch_job.model_dump())

    def save_batch_output(self, batch_id: Optional[str] = None, output_path: Optional[Path] = None):
        """Save the output of a batch job to a file.

        Args:
            batch_id (Optional[str]): The ID of the batch job. If not provided, uses the last batch ID.
            output_path (Optional[Path]): The path to save the output file. If not provided, saves to the current directory.
        """
        batch_id = self._check_batch_id(batch_id)

        file_response = self.get_batch_output(batch_id)

        if output_path is None:
            output_path = Path.cwd() / f"output_{batch_id}.joblib"
        else:
            output_path = Path(output_path)

        assert output_path.suffix == '.joblib', "Output file must be in joblib format"

        joblib.dump(file_response, output_path)

        print(f"Batch output saved to {output_path}")
        
    def load_batch_output(self, output_path: Path) -> List[BatchOutput]:
        """Load the output of a batch job from a file.

        Args:
            output_path (Path): The path to the output file.

        Returns:
            List[BatchOutput]: The output of the batch job.
        """
        assert output_path.suffix == '.joblib', "Output file must be in joblib format"
        assert output_path.exists(), f"Output file {output_path} not found"

        return joblib.load(output_path)    
        
    def _check_batch_id(self, batch_id: Optional[str] = None):
        if batch_id is None:
            if self.last_batch is None:
                raise ValueError("No batch_id provided and no last_batch available.")
            return self.last_batch.id
        return batch_id
    
    def calculate_batch_price(self, price_per_milion_input: float, price_per_milion_output:float, batch_id: Optional[str] = None) -> float:
        """Calculate the price of a batch job.

        Args:
            price_per_milion_input (float): The price per million tokens for input.
            price_per_milion_output (float): The price per million tokens for output.
            batch_id (Optional[str]): The ID of the batch job. If not provided, uses the last batch ID.

        Returns:
            float: The total price of the batch job.
        """

        batch_id = self._check_batch_id(batch_id)
        batch_output = self.get_batch_output(batch_id)
        batch_output = [json.loads(output) for output in batch_output]
        total_input_tokens = sum([i['response']['body']['usage']['prompt_tokens'] for i in batch_output])
        total_output_tokens = sum([i['response']['body']['usage']['completion_tokens'] for i in batch_output])

        total_price = (total_input_tokens * price_per_milion_input / 1_000_000) + (total_output_tokens * price_per_milion_output / 1_000_000)
        return f'USD ${total_price:.2f}'
        