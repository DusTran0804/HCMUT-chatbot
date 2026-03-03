#!/bin/bash

cd "$(dirname "$0")"
source venv/bin/activate
python ingest.py sample_data.txt

venv/bin/python chatbot.py
