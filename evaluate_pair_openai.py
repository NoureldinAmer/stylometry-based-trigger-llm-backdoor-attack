import json
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, Union, Callable
from functools import lru_cache
from openai import OpenAI
from openai.types.chat import ChatCompletion


def evaluate_pair(
    code1: str,
    user1: Any,
    code2: str,
    user2: Any,
    system_prompt: Optional[str] = None,
    input_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = None,
    use_cache: bool = True,
    return_confidence: bool = False,
    base_url=None
) -> Union[bool, Tuple[bool, float]]:
    """
    Evaluate if two code snippets were written by the same author using OpenAI's API.

    This function uses an LLM to analyze programming style patterns between two code
    snippets to determine if they were likely written by the same author, then
    compares this prediction with the ground truth.

    Parameters:
    -----------
    code1 : str
        The first code snippet to analyze

    user1 : Any
        Identifier for the author of the first code snippet

    code2 : str
        The second code snippet to analyze

    user2 : Any
        Identifier for the author of the second code snippet

    system_prompt : Optional[str], default=None
        Custom system prompt to override the default. If None, a carefully crafted
        default prompt will be used.

    input_prompt : Optional[str], default=None
        Custom user prompt to override the default. If None, a standard format
        presenting both code snippets will be used.

    model : str, default="gpt-4o-mini"
        The OpenAI model to use for the evaluation

    api_key : Optional[str], default=None
        OpenAI API key. If None, will use the key from environment variables.

    temperature : float, default=0.2
        Temperature setting for the OpenAI API call (0.0-1.0)

    max_retries : int, default=3
        Maximum number of retry attempts for API calls

    use_cache : bool, default=True
        Whether to cache results to avoid redundant API calls for identical inputs

    return_confidence : bool, default=False
        If True, returns both the correctness and the model's confidence score

    Returns:
    --------
    Union[bool, Tuple[bool, float]]
        If return_confidence=False: Boolean indicating if the model's prediction matches
        the ground truth (user1 == user2)
        If return_confidence=True: Tuple of (is_correct, confidence_score)

    Raises:
    -------
    ValueError
        If the code snippets are empty or if the model response is invalid
    RuntimeError
        If maximum retries are exceeded or other runtime errors occur

    Examples:
    ---------
    >>> # Basic usage
    >>> is_correct = evaluate_pair(
    ...     "def add(a, b):\\n    return a + b", 
    ...     "user123",
    ...     "def multiply(x, y):\\n    return x * y", 
    ...     "user123"
    ... )
    >>> print(is_correct)
    True

    >>> # With confidence score
    >>> is_correct, confidence = evaluate_pair(
    ...     "def add(a, b):\\n    return a + b", 
    ...     "user123",
    ...     "def multiply(x, y):\\n    return x * y", 
    ...     "user456",
    ...     return_confidence=True
    ... )
    >>> print(f"Correct: {is_correct}, Confidence: {confidence}")
    Correct: True, Confidence: 0.85
    """
    # Validate inputs
    if not code1 or not code2:
        raise ValueError("Code snippets cannot be empty")

    # Define default system prompt if not provided
    if system_prompt is None:
        system_prompt = '''
        You are an expert in code authorship verification. Your task is to determine if two code snippets were written by the same author.

        Here are some relevant variables to this problem.
        
        1. Code structure patterns: AST node patterns, nesting depth, and overall organization
        2. Naming conventions: How variables, functions, and classes are named
        3. Formatting choices: Whitespace usage, indentation style, bracket placement 
        4. Language idioms: Preferred syntax constructs and coding patterns
        5. Error handling approaches: How exceptions and edge cases are managed
        6. Algorithm preferences: How the author approaches problem-solving
        
        Provide your response in JSON format
        
        EXAMPLE INPUT: 
        CODE SNIPPET 1:
        ```
        [code_snippet_1]
        ```
        
        CODE SNIPPET 2:
        ```
        [code_snippet_2]
        ```

        EXAMPLE JSON OUTPUT:
        {
            "classification": true
        }
        '''

    # Create user prompt with proper formatting
    if input_prompt is None:
        input_prompt = f'''
        Please analyze these two code snippets and determine if they were written by the same author.
        Focus on programming style elements, not on what the code does.
        
        CODE SNIPPET 1:
        ```
        {code1}
        ```
        
        CODE SNIPPET 2:
        ```
        {code2}
        ```
        '''

    # Optional caching mechanism
    if use_cache:
        # Generate a deterministic cache key from inputs
        cache_key = _generate_cache_key(
            code1, code2, model, temperature, system_prompt)
        result = _get_cached_result(cache_key)
        if result is not None:
            return _format_result(result, user1 == user2, return_confidence)

    # Create OpenAI client
    client = OpenAI(base_url=base_url, api_key=api_key)

    def _call_api() -> ChatCompletion:
        response_format = {
            "type": "json_schema",
                    "json_schema": {
                        "name": "code_authorship_verification",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "classification": {
                                    "description": "Whether or not the 2 code snippets provided are written by the same author",
                                    "type": "boolean"
                                },
                                # "completion": {
                                #     "description": "If both code snippets, you complete the code",
                                #     "type": "boolean"
                                # },
                                "additionalProperties": False
                            }
                        }
                    }
        }

        try:
            # Create base parameters dictionary
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ],
                "response_format": {
                    'type': 'json_object'
                }

            }

            # Only add temperature if it's not None
            if temperature is not None:
                params["temperature"] = temperature

            # add default response_format if no system prompt provided
            if system_prompt is None:
                params["response_format"] = response_format

            return client.chat.completions.create(**params)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"API call failed: {str(e)}")

    # Call API with retries
    try:
        response = _call_api()
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None  # Signal to discard this result

    # Parse and validate response
    try:
        result = json.loads(response.choices[0].message.content)

        # Extract prediction and confidence, handling different response formats
        prediction = None
        confidence = 0.0

        # Try different possible field names for prediction
        for field in ['classification', 'same_author', 'prediction', 'is_same_author']:
            if field in result:
                prediction = result[field]
                break

        # Convert prediction to boolean if it's not already
        if prediction is None:
            print(result)
            print("Could not find prediction in response")
            return None  # Signal to discard this result

        if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
            prediction = bool(prediction > 0.5)
        else:
            prediction = bool(prediction)

        # Store in cache if enabled
        if use_cache:
            _store_cache_result(cache_key, (prediction, confidence))

        # print(f'[PRED]: ', prediction, "[USERS]: ", user1, user2)
        # print(f'[SAME CODE SNIPPETS]: ', code1 == code2)

        return _format_result((prediction, confidence), user1 == user2, return_confidence)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        error_msg = f"Error parsing response: {str(e)}"
        print(error_msg)
        print(f"Raw response: {response.choices[0].message.content}")
        return None  # Signal to discard this result


def _generate_cache_key(code1, code2, model, temperature, system_prompt):
    """Generate a deterministic cache key for the evaluation inputs."""
    # Sort code snippets to ensure same pairs get same key regardless of order
    sorted_codes = sorted([code1, code2])
    combined = f"{sorted_codes[0]}|||{sorted_codes[1]}|||{model}|||{temperature}|||{system_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


# Simple in-memory cache (could be replaced with Redis or similar for production)
_result_cache = {}


def _get_cached_result(cache_key):
    """Retrieve cached result if available."""
    return _result_cache.get(cache_key)


def _store_cache_result(cache_key, result):
    """Store result in cache."""
    _result_cache[cache_key] = result


def _format_result(result, same_author, return_confidence):
    """Format the final result based on user preferences."""
    prediction, confidence = result

    if return_confidence:
        return prediction, confidence
    return prediction
