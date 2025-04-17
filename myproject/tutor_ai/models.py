from django.db import models
import uuid  # For generating unique session IDs
from django.contrib.auth.models import User
from enum import Enum

# Existing model: Session logging
class TutoringSession(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False, unique=True)
    user_id = models.CharField(max_length=100, default="guest")
    task_difficulty = models.FloatField()
    error_rate = models.FloatField()
    response_time = models.FloatField()
    rl_decision = models.CharField(max_length=50)
    model_confidence = models.FloatField(default=1.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    used_hint= models.BooleanField(default=False)
    took_break = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"Session {self.session_id} | User: {self.user_id} | Decision: {self.rl_decision} | {self.timestamp}"

# Existing model: Lesson content (unchanged)
class LessonContent(models.Model):
    content_type = models.CharField(max_length=50)  # e.g., "hint", "example"
    content_text = models.TextField()

# 🔥 NEW model: Tracks historical performance across sessions
class CognitiveStateEnum(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    average_response_time = models.FloatField(default=0.0)
    average_error_rate = models.FloatField(default=0.0)
    tasks_completed = models.IntegerField(default=0)

    hints_used = models.IntegerField(default=0)
    breaks_taken = models.IntegerField(default=0)

    initial_diagnostic_state = models.CharField(
        max_length=10,
        choices=[(tag.value, tag.value) for tag in CognitiveStateEnum],
        null=True, blank=True
    )

    last_cognitive_load_prediction = models.CharField(
        max_length=10,
        choices=[(tag.value, tag.value) for tag in CognitiveStateEnum],
        null=True, blank=True
    )

    def __str__(self):
        return f"{self.user.username}'s Profile"
