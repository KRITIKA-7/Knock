from pydantic import ValidationError
from backend.app.schemas import ResumeProfile
from backend.app.services.llm_client import call_llm, invalidate_llm_cache

SYSTEM_PROMPT = "You extract structured data from resume text and return ONLY valid JSON. Never include markdown, code fences, or commentary."

USER_PROMPT_TEMPLATE = """Parse the resume text below into a single JSON object matching this exact schema:

{{
  "name": string | null,
  "skills": [string],
  "education": [
    {{
      "institution": string,
      "degree": string | null,
      "year": string | null
    }}
  ],
  "experience": [
    {{
      "company": string,
      "role": string,
      "duration": string | null,
      "claims": [
        {{
          "text": string,
          "type": "technology" | "numeric_metric" | "feature" | "deployment" | "timeline" | "other",
          "source_excerpt": string,
          "ambiguous": boolean
        }}
      ],
      "links": [string]
    }}
  ],
  "projects": [
    {{
      "name": string,
      "description": string,
      "technologies": [string],
      "claims": [
        {{
          "text": string,
          "type": "technology" | "numeric_metric" | "feature" | "deployment" | "timeline" | "other",
          "source_excerpt": string,
          "ambiguous": boolean
        }}
      ],
      "links": [string]
    }}
  ]
}}

Rules:
- Return ONLY JSON matching the schema.
- Extract only what is in the text; never invent skills, claims, or links.
- Do NOT extract phone number or email.
- Rejoin words broken across lines by hyphens.
- Each Claim.text MUST contain strictly ONE atomic, checkable fact: one technology, one numeric metric, one feature, one deployment target, or one timeline. NEVER combine comma-separated or "and"-joined lists of technologies, features, or metrics into one Claim.text.
  Example:
  Bullet: "Built X with React and FastAPI featuring RBAC and audit logs (95% accuracy, 0.91 F1)."
  Split into separate claims (each pointing to the verbatim source_excerpt):
  - {{"text": "Built X using React", "type": "technology", ...}}
  - {{"text": "Built X using FastAPI", "type": "technology", ...}}
  - {{"text": "Implemented RBAC in X", "type": "feature", ...}}
  - {{"text": "Implemented audit logs in X", "type": "feature", ...}}
  - {{"text": "Achieved 95% accuracy in X", "type": "numeric_metric", ...}}
  - {{"text": "Achieved 0.91 F1 in X", "type": "numeric_metric", ...}}
- Every Claim must include "source_excerpt": the exact verbatim substring from the resume that the claim came from.
- Copy numbers and units exactly as written. If wording is ambiguous (unclear baseline, unit, or scope), keep the original phrasing in "text" and set "ambiguous": true on the Claim; otherwise set "ambiguous": false.
- Time-based claims (such as "in 1 day") must have type "timeline".
- A project that has its own repo link or its own name (such as a hackathon project mentioned inside an internship bullet) must be a separate entry under "projects", NOT nested inside "experience". Experience claims must only include work actually described for the employer.
- Capture every GitHub URL and live URL under the project or experience item it belongs to.

Resume text:
{text}"""


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def parse_resume(text: str) -> ResumeProfile:
    if not text or not text.strip():
        raise ValueError("Resume text is empty")

    prompt = USER_PROMPT_TEMPLATE.format(text=text)
    raw = call_llm(prompt, system=SYSTEM_PROMPT)

    try:
        return ResumeProfile.model_validate_json(_strip_code_fences(raw))
    except (ValidationError, ValueError) as first_err:
        invalidate_llm_cache(prompt, system=SYSTEM_PROMPT)
        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Validation failed with error:\n{first_err}\n\n"
                    "Return ONLY valid JSON matching the schema with no code fences or extra text."
                ),
            },
        ]
        retry_raw = call_llm(retry_messages, system=SYSTEM_PROMPT)
        try:
            return ResumeProfile.model_validate_json(_strip_code_fences(retry_raw))
        except (ValidationError, ValueError) as second_err:
            invalidate_llm_cache(retry_messages, system=SYSTEM_PROMPT)
            raise ValueError(f"Failed to parse resume after retry: {second_err}") from second_err
