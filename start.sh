#!/bin/zsh
cd ~/Desktop/dental_bot
source venv/bin/activate
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
uvicorn app:app --port 8000
