from django.urls import path
from .views import tutoring_decision, get_user_sessions, session_history

urlpatterns = [
    path("api/rl_decision/", tutoring_decision, name="tutoring_decision"),
    path("api/get_sessions/", get_user_sessions, name="get_sessions"),
    path("session_history/", session_history, name="session_history"),
]
    