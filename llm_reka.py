import click
import llm
import reka


MODELS = ("reka-core", "reka-edge", "reka-flash")


@llm.hookimpl
def register_models(register):
    for model_id in MODELS:
        register(
            Reka(model_id),
        )


class Reka(llm.Model):
    needs_key = "reka"

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        reka.API_KEY = llm.get_key("", "reka", "LLM_REKA_KEY")
        conversation_history = []
        if conversation:
            for response in conversation.responses:
                conversation_history.append(
                    {
                        "type": "human",
                        "text": response.prompt.prompt,
                    }
                )
                conversation_history.append({"type": "model", "text": response.text()})
        response = reka.chat(
            human=prompt.prompt,
            conversation_history=conversation_history,
        )
        yield response["text"]

    def __str__(self):
        return "Reka: {}".format(self.model_id)
