from smolagents import CodeAgent, tool, load_tool
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import datetime, pytz, yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

# ----------------- Tools -----------------
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """A custom tool that simply returns its input arguments.

    Args:
        arg1 (str): The first argument
        arg2 (int): The second argument

    Returns:
        str: A formatted string showing the received arguments
    """
    return f"My tool got: {arg1} and {arg2}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Return the current time in a given timezone.

    Args:
        timezone (str): A valid timezone string (e.g., 'America/New_York')

    Returns:
        str: The current local time in the specified timezone, or an error message if invalid
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------- Final Answer Tool -----------------
final_answer = FinalAnswerTool()

# ----------------- Prompt Templates -----------------
with open("testPrompts.yaml") as f:
    prompt_templates = yaml.safe_load(f)

# ----------------- Local Hugging Face Model Wrapper -----------------
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class LocalHFWrapper:
    def __init__(self, model, tokenizer, max_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.last_input_token_count = 0

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_tokens
        )

    def __call__(self, prompt, **kwargs):
        # Ensure prompt is a string
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        self.last_input_token_count = len(prompt.split())

        # Generate text
        output = self.generator(prompt, max_new_tokens=self.max_tokens)[0]["generated_text"]
        return output

wrapped_model = LocalHFWrapper(model, tokenizer)

# ----------------- Load Other Tools -----------------
image_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# ----------------- Create Agent -----------------
agent = CodeAgent(
    model=wrapped_model,
    tools=[final_answer, get_current_time_in_timezone, my_custom_tool, image_tool],
    max_steps=6,
    verbosity_level=1,
    prompt_templates=prompt_templates
)

# ----------------- Launch Gradio -----------------
GradioUI(agent).launch()
