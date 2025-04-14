from .models import UserProfile

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
