# Orbit Voice Chat (with Sesame CSM)

This app uses the `sesame/CSM` conversational model to extract audio embeddings from speech and generate GPT-4 replies.

## Features

- Upload `.wav` file in browser
- Get contextual embeddings via CSMModel
- Get GPT-4 response
- Runs on Streamlit Cloud

## Setup

1. Clone the repo
2. Add your OpenAI key to `.env`
3. Run with `streamlit run app/voice_chat_app.py`

## Deployment

Deploy via Streamlit Cloud and set `OPENAI_API_KEY` in the Secrets tab.