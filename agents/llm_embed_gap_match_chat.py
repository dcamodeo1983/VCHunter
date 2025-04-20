import openai

class ChatbotAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def create(self, vc_summaries, founder_text):
        context = "\n\n".join(vc_summaries)
        return SimpleChatResponder(context, self.api_key)


class SimpleChatResponder:
    def __init__(self, context, api_key):
        self.context = context
        openai.api_key = api_key

    def ask(self, question):
        try:
            messages = [
                {"role": "system", "content": "You are an expert VC analyst. Use the context to answer the user's question clearly."},
                {"role": "user", "content": f"Context:\n{self.context}\n\nQuestion: {question}"}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Error: {e}"
