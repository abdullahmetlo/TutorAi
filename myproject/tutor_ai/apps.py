from django.apps import AppConfig

class TutorAiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tutor_ai'

    def ready(self):
        import tutor_ai.signals 
