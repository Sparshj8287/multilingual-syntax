import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key="AIzaSyAeTEXENYDFTf5oeTUK0rZwN1G4NAVt3O0",
    )

    model = "gemini-3-pro-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""How to make a sandwich?"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()


