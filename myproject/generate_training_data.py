import os
import django
from pathlib import Path
import pandas as pd

# ✅ Step 1: Setup Django
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

# ✅ Step 2: Import after Django is configured
from tutor_ai.models import TutoringSession, UserProfile
from django.contrib.auth.models import User
from tutor_ai.rl_model import estimate_cognitive_load

# ✅ Step 3: Build dataset
data = []

for session in TutoringSession.objects.all():
    try:
        user = User.objects.get(username=session.user_id)
        profile = user.userprofile
    except:
        profile = None

    features = {
        "task_difficulty": session.task_difficulty,
        "error_rate": session.error_rate,
        "response_time": session.response_time,
        "avg_error_rate": profile.average_error_rate if profile else 0.5,         # ✅ default fallback
        "avg_response_time": profile.average_response_time if profile else 30.0,  # ✅ default fallback
    }

    # Use heuristic to estimate label (for now)
    load_state = estimate_cognitive_load(
        session.task_difficulty,
        session.error_rate,
        session.response_time
    )
    features["label"] = load_state.name  # LOW / MEDIUM / HIGH

    data.append(features)
    print(f"✅ Added row: {features}")  # ✅ Debugging log

# ✅ Convert to DataFrame
df = pd.DataFrame(data)

# ✅ Sanity check: Drop rows only if core features are missing
df = df.dropna(subset=["task_difficulty", "error_rate", "response_time"])

# ✅ Debug preview
print("\n👀 Data Preview:")
print(df.head())

# ✅ Row count check
print(f"\n🔍 Final dataset size: {df.shape[0]} rows")

# ✅ Save to CSV
df.to_csv("cognitive_load_dataset.csv", index=False)
print("✅ Dataset saved as cognitive_load_dataset.csv")
