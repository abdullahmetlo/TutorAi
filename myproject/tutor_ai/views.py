import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.shortcuts import render
from .rl_model import (
    load_q_table,
    get_rl_recommendation,
    ml_estimate_cognitive_load,  # ✅ NEW ML-based function
    update_transition_counts
)
from .models import TutoringSession
from .utils import update_user_profile
from django.contrib.auth.models import User
import uuid
import numpy as np

q_table = load_q_table()

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

            # ✅ Determine average values (for ML input)
            avg_error = 0.5
            avg_time = 30.0
            if request.user.is_authenticated and hasattr(request.user, "userprofile"):
                avg_error = request.user.userprofile.average_error_rate
                avg_time = request.user.userprofile.average_response_time

            # ✅ Use ML-based cognitive load estimation
            load_state = ml_estimate_cognitive_load(
                task_difficulty=data["task_difficulty"],
                error_rate=data["error_rate"],
                response_time=data["response_time"],
                avg_error_rate=avg_error,
                avg_response_time=avg_time
            )
            state_index = load_state.value

            # Get recommendation
            recommendation = get_rl_recommendation(data, q_table)

            # Save session
            session = TutoringSession.objects.create(
                session_id=session_id,
                user_id=user_id,
                task_difficulty=data["task_difficulty"],
                error_rate=data["error_rate"],
                response_time=data["response_time"],
                rl_decision=recommendation["decision"]
            )
            session.save()

            # Try to update user profile using Django user ID
            try:
                user_obj = User.objects.get(username=user_id)
                update_user_profile(
                    user=user_obj,
                    response_time=data["response_time"],
                    is_correct=(data["error_rate"] == 0),
                    hint_used=data.get("hint_used", False),
                    break_taken=(recommendation["decision"] == "Introduce Break")
                )
            except User.DoesNotExist:
                pass  # Guest or user not found

            # Optional: Update transition model
            update_transition_counts(state_index, state_index, np.argmax(q_table[state_index]))

            return JsonResponse({
                "session_id": str(session.session_id),
                "user_id": user_id,
                "decision": recommendation["decision"],
                "next_content": recommendation["next_content"],
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
