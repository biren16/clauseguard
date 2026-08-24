import json
import os

from dotenv import load_dotenv

from modules.evidence_model import GroqEvidenceModel


load_dotenv(override=True)


assert os.environ.get("GROQ_API_KEY"), (
    "GROQ_API_KEY is not loaded"
)


model = GroqEvidenceModel()


response = model.generate(
    system_prompt=(
        "Return only a JSON object with one field called "
        '"status". The value must be "OK".'
    ),
    user_prompt="Are you working?",
    json_mode=True,
)


parsed = json.loads(response)


assert parsed == {"status": "OK"}


print("GROQ PROVIDER TEST: PASSED")