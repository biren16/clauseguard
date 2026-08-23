import os

from dotenv import load_dotenv

from modules.evidence_model import GroqEvidenceModel


load_dotenv(override=True)

print(
    "GROQ key loaded:",
    bool(os.environ.get("GROQ_API_KEY")),
)

model = GroqEvidenceModel()

response = model.generate(
    system_prompt=(
        "Return only a JSON object with one field called "
        '"status". The value must be "OK".'
    ),
    user_prompt="Are you working?",
)

print("GROQ RESPONSE:")
print(response)