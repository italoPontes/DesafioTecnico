OCR_PROMPT = """
You are an OCR engine.

Read every visible character from the document.

Rules:

- Do not summarize.
- Do not correct spelling.
- Do not infer missing text.
- Preserve line breaks.
- Preserve punctuation.
- Ignore the background.

Return ONLY valid JSON.

{
"document": {
    "type": "identity_card",
    "country": "BR",
    "issuer": {
    "country": "",
    "government": "",
    "state": "",
    "agency": ""
    },
    "title": "Carteira de Identidade"
},
"fields": {
    "name": {
    "label": "Nome / Name",
    "value": ""
    },
    "social_name": {
    "label": "Nome Social / Social Name",
    "value": ""
    },
    "personal_number": {
    "label": "Registro Geral - CPF / Personal Number",
    "value": ""
    },
    "sex": {
    "label": "Sexo / Sex",
    "value": ""
    },
    "birth_date": {
    "label": "Data de Nascimento / Date of Birth",
    "value": ""
    },
    "nationality": {
    "label": "Nacionalidade / Nationality",
    "value": ""
    },
    "place_of_birth": {
    "label": "Naturalidade / Place of Birth",
    "value": ""
    },
    "expiry_date": {
    "label": "Data de Validade / Date of Expiry",
    "value": ""
    }
}
}
"""