import llm
import httpx
from httpx_sse import connect_sse
import json


MODELS = ("reka-core", "reka-edge", "reka-flash")


@llm.hookimpl
def register_models(register):
    for model_id in MODELS:
        register(
            Reka(model_id),
        )


class Reka(llm.Model):
    needs_key = "reka"
    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        with httpx.Client() as client:
            with connect_sse(
                client,
                "POST",
                "https://api.reka.ai/chat",
                headers={
                    "x-api-key": llm.get_key("", "reka", "LLM_REKA_KEY"),
                },
                json={
                    "conversation_history": [
                        {
                            "type": "human",
                            "text": prompt.prompt,
                        }
                    ],
                    "use_search_engine": False,
                    "use_code_interpreter": False,
                    "model_name": self.model_id,
                    "stream": True,
                    # "random_seed": 1713403859366
                },
            ) as event_source:
                last_text = ""
                accumulated = []
                for sse in event_source.iter_sse():
                    accumulated.append(sse.json())
                    info = json.loads(sse.data)
                    text = info["text"]
                    if text != last_text:
                        # Figure out what's new and yield that
                        new_text = text[len(last_text) :]
                        yield new_text
                        last_text = text
                print()
                print(json.dumps(accumulated, indent=2))
