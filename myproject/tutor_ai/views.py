import json
import uuid
import numpy as np
import logging
import time

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.shortcuts import render
from django.contrib.auth.models import User
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_GET
from django.http import JsonResponse

from .models import TutoringSession
from .utils import (
    update_user_profile,
    get_oer_content,
    simplify_text_with_ai,
    estimate_initial_load,
    load_diagnostic_questions
)
from .rl_model import (
    load_q_table,
    get_rl_recommendation,
    ml_estimate_cognitive_load,
    update_transition_counts
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

q_table = load_q_table()

@csrf_exempt
def diagnostic_view(request):
    questions = load_diagnostic_questions()
    return render(request, "diagnostic.html", {"questions": questions})

@csrf_exempt
@require_http_methods(["POST"])
def submit_diagnostic(request):
    try:
        data = json.loads(request.body)
        answers = data.get("answers", [])
        response_time = data.get("response_time", 30.0)
        used_hint = data.get("used_hint", False)
        took_break = data.get("took_break", False)

        logger.info(f"Diagnostic POST data: {data}")
        logger.info(f"Diagnostic submitted. Hint used: {used_hint}, Took break: {took_break}")

        if not answers:
            return JsonResponse({"error": "No answers provided."}, status=400)

        cognitive_load = estimate_initial_load(answers, response_time, used_hint, took_break)

        return JsonResponse({
            "cognitive_load": cognitive_load
        }, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def diagnostic_submit(request):
    if request.method == "POST":
        data = json.loads(request.body)
        answers = data.get("answers", [])
        response_time = data.get("response_time", 30.0)
        used_hint = data.get("used_hint", False)
        took_break = data.get("took_break", False)

        if not answers:
            return JsonResponse({"error": "No answers submitted."}, status=400)

        load = estimate_initial_load(answers, response_time, used_hint, took_break)

        return JsonResponse({
            "message": "Initial cognitive load estimated successfully.",
            "initial_load": load
        })
    return JsonResponse({"error": "Only POST requests allowed."}, status=405)

@csrf_exempt
def tutoring_decision(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            required_fields = ["task_difficulty", "error_rate", "response_time"]
            if any(field not in data for field in required_fields):
                return JsonResponse({"error": "Missing required fields"}, status=400)

            user_id = data.get("user_id", "guest")
            session_id = uuid.uuid4()
            used_hint = data.get("hint_used", False)
            took_break = data.get("took_break", False)
            cognitive_load = data.get("cognitive_load", "unknown")  # ✅ Step 4

            # ML-based cognitive load estimation with fallback values
            avg_error = 0.5
            avg_time = 30.0
            if request.user.is_authenticated and hasattr(request.user, "userprofile"):
                avg_error = request.user.userprofile.average_error_rate
                avg_time = request.user.userprofile.average_response_time

            load_state = ml_estimate_cognitive_load(
                task_difficulty=data["task_difficulty"],
                error_rate=data["error_rate"],
                response_time=data["response_time"],
                avg_error_rate=avg_error,
                avg_response_time=avg_time
            )
            state_index = load_state.value

            state_features = {
                "task_difficulty": data["task_difficulty"],
                "error_rate": data["error_rate"],
                "response_time": data["response_time"],
                "used_hint": used_hint,
                "took_break": took_break,
            }

            # RL decision making
            recommendation = get_rl_recommendation(data, q_table)

            # Fetch OER content
            oer_content = get_oer_content(recommendation["decision"])
            raw_text = (
                oer_content.get("explanation")
                or oer_content.get("hint")
                or oer_content.get("example")
                or oer_content.get("note", "No content found.")
            )

            try:
                simplified = simplify_text_with_ai(raw_text)
            except Exception as e:
                logger.error(f"Hugging Face API error: {e}")
                simplified = f"(AI Error) {raw_text}"

            # ✅ Save session including cognitive_load
            session = TutoringSession.objects.create(
                session_id=session_id,
                user_id=user_id,
                task_difficulty=data["task_difficulty"],
                error_rate=data["error_rate"],
                response_time=data["response_time"],
                rl_decision=recommendation["decision"],
                used_hint=used_hint,
                took_break=took_break,
                cognitive_load=cognitive_load  # ✅ STEP 4 saved here
            )

            try:
                user_obj = User.objects.get(username=user_id)
                update_user_profile(
                    user=user_obj,
                    response_time=data["response_time"],
                    is_correct=(data["error_rate"] == 0),
                    hint_used=used_hint,
                    break_taken=took_break
                )
            except User.DoesNotExist:
                pass

            update_transition_counts(state_index, state_index, np.argmax(q_table[state_index]))

            return JsonResponse({
                "session_id": str(session.session_id),
                "user_id": user_id,
                "decision": recommendation["decision"],
                "next_content": simplified,
                "message": "Session saved successfully"
            }, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

def get_user_sessions(request):
    if request.method == "GET":
        user_id = request.GET.get("user_id")
        sessions = TutoringSession.objects.filter(user_id=user_id).order_by("-timestamp") if user_id else TutoringSession.objects.all().order_by("-timestamp")
        session_data = [
            {
                "session_id": str(session.session_id),
                "user_id": session.user_id,
                "task_difficulty": session.task_difficulty,
                "error_rate": session.error_rate,
                "response_time": session.response_time,
                "rl_decision": session.rl_decision,
                "used_hint": session.used_hint,
                "took_break": session.took_break,
                "cognitive_load": session.cognitive_load,
                "timestamp": session.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for session in sessions
        ]
        return JsonResponse({"sessions": session_data}, status=200)

    return JsonResponse({"error": "Only GET requests are allowed"}, status=405)

def session_history(request):
    user_id = request.GET.get("user_id", None)
    sessions = TutoringSession.objects.filter(user_id=user_id).order_by("-timestamp") if user_id else TutoringSession.objects.all().order_by("-timestamp")

    paginator = Paginator(sessions, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "session_history.html", {
        "page_obj": page_obj,
        "user_id": user_id,
        "message": "No session history found." if not page_obj.object_list else "",
    })

def tutor_ui(request):
    return render(request, "tutor_ui.html")

@require_GET
def get_tutoring_question(request):
    # Placeholder logic – later we can use RL or cognitive load to personalize this
    question = {
        "question": "What is 3/4 + 1/8?",
        "options": ["7/8", "1/2", "1", "5/8"],
        "correctIndex": 0,
        "hint": "Find a common denominator."
    }
    return JsonResponse(question)

@csrf_exempt
@require_http_methods(["POST"])
def log_tutoring_response(request):
    try:
        data = json.loads(request.body)

        session_id = data.get("session_id", str(uuid.uuid4()))
        user_id = data.get("user_id", "guest")
        question_id = data.get("question_id", "N/A")
        selected_index = data.get("selected_index")
        is_correct = data.get("is_correct", False)
        response_time = float(data.get("response_time", 0.0))
        cognitive_load = data.get("cognitive_load", "unknown")
        used_hint = data.get("used_hint", False)
        took_break = data.get("took_break", False)

        TutoringSession.objects.create(
            session_id=session_id,
            user_id=user_id,
            task_difficulty="unknown",
            error_rate=0 if is_correct else 1,
            response_time=response_time,
            rl_decision="auto",
            used_hint=used_hint,
            took_break=took_break,
            cognitive_load=cognitive_load,
        )

        return JsonResponse({"message": "Response logged"}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


