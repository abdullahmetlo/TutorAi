import json
import os
from .models import UserProfile
from dotenv import load_dotenv
import requests
import logging
from django.conf import settings

def load_diagnostic_questions():
    file_path = os.path.join(settings.BASE_DIR, 'tutor_ai', 'data', 'diagnostic_questions.json')
    with open(file_path, 'r') as file:
        return json.load(file)

def estimate_initial_load(answers, response_time, used_hint=False, took_break=False):
    correct_answers = sum(1 for ans in answers if ans.get("correct"))
    avg_time = response_time / len(answers)

    # Basic rules (adjust as needed):
    if correct_answers >= len(answers) * 0.8 and not used_hint and not took_break:
        return "low"
    elif correct_answers >= len(answers) * 0.5 or avg_time > 20:
        return "medium"
    else:
        return "high"

#Existing user profile updater
def update_user_profile(user, response_time, is_correct, hint_used=False, break_taken=False):
    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return  # Skip update if no profile

    total_tasks = profile.tasks_completed + 1

    profile.average_response_time = (
        (profile.average_response_time * profile.tasks_completed) + response_time
    ) / total_tasks

    profile.average_error_rate = (
        (profile.average_error_rate * profile.tasks_completed) + int(not is_correct)
    ) / total_tasks

    profile.tasks_completed = total_tasks

    if hint_used:
        profile.hints_used += 1
    if break_taken:
        profile.breaks_taken += 1

    profile.save()


#Fetch OER content based on strategy and topic
def get_oer_content(strategy, topic="fractions"):
    try:
        file_path = os.path.join(os.path.dirname(__file__), "data", "fraction_questions.json")
        with open(file_path, "r") as f:
            data = json.load(f)

        # Filter by topic + strategy
        matches = [item for item in data if item["strategy"] == strategy and item["topic"] == topic]

        return matches[0] if matches else {"note": "No matching OER content found."}
    except Exception as e:
        return {"error": str(e)}
    
load_dotenv()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")

def simplify_text_with_ai(text):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 100,
            "do_sample": False
        }
    }

    try:
        logger.info(f"Sending request to HF with text: {text}")
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        logger.info(f"HF Response Status: {response.status_code}")
        logger.info(f"HF Response Body: {response.text}")

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                logger.warning("Unexpected HF response format")
                return f"(Unexpected HF response) {text}"
        else:
            logger.error(f"Hugging Face returned an error: {response.status_code}")
            return f"(AI Error) {text}"

    except Exception as e:
        logger.exception("Exception while calling Hugging Face API")
        return f"(AI simplification failed) {text}"