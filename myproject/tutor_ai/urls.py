from django.urls import path
from .views import tutoring_decision, get_user_sessions, session_history, tutor_ui, diagnostic_view, submit_diagnostic, get_tutoring_question

urlpatterns = [
    path("rl_decision/", tutoring_decision, name="tutoring_decision"),
    path("get_sessions/", get_user_sessions, name="get_sessions"),
    path("session_history/", session_history, name="session_history"),
    path("tutor/", tutor_ui, name="tutor_ui"),
    path("diagnostic/", diagnostic_view, name="diagnostic_view"),
    path("submit_diagnostic/", submit_diagnostic, name="submit_diagnostic"),
    path("api/get_tutoring_question/", get_tutoring_question, name="get_tutoring_question"),
    path("api/log_tutoring_response/", log_tutoring_response, name="log_tutoring_response"),


]