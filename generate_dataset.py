"""
FitMatch - Dataset Generation Module
Generates simulated users, workout plans, and ratings for the recommender system.
"""

import numpy as np
import pandas as pd
import os

# ============================================================
# Configuration
# ============================================================
SEED = 42
NUM_USERS = 100
NUM_PLANS = 60
MIN_RATINGS = 850  # Target minimum number of ratings
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(SEED)

# ============================================================
# 1. Generate Users
# ============================================================
def generate_users(n=NUM_USERS):
    """Generate n users with demographic and fitness attributes."""
    fitness_levels = ["beginner", "intermediate", "advanced"]
    goals = ["weight_loss", "muscle_gain", "endurance", "flexibility"]

    users = pd.DataFrame({
        "user_id": range(1, n + 1),
        "age": np.random.randint(18, 60, size=n),
        "fitness_level": np.random.choice(fitness_levels, size=n, p=[0.4, 0.35, 0.25]),
        "goal": np.random.choice(goals, size=n, p=[0.30, 0.30, 0.25, 0.15]),
    })
    return users


# ============================================================
# 2. Generate Workout Plans
# ============================================================
def generate_workout_plans(n=NUM_PLANS):
    """Generate n workout plans with type, difficulty, duration, target goal, and description."""

    plan_templates = [
        # Cardio plans
        ("Morning Run Blast", "cardio", "beginner", 30, "weight_loss",
         "A brisk morning running routine focused on burning calories and boosting metabolism through steady-state cardio."),
        ("Sprint Interval Training", "cardio", "advanced", 25, "endurance",
         "High-intensity sprint intervals alternating with recovery jogs to build explosive speed and cardiovascular endurance."),
        ("Fat Burn Cycling", "cardio", "intermediate", 40, "weight_loss",
         "Indoor cycling session with varying resistance levels designed for maximum fat oxidation and calorie burn."),
        ("Jump Rope Cardio", "cardio", "intermediate", 20, "weight_loss",
         "Fast-paced jump rope workout combining single and double unders for cardio conditioning and coordination."),
        ("Stairclimber Endurance", "cardio", "beginner", 35, "endurance",
         "Steady stairclimber workout building lower body endurance and cardiovascular fitness at moderate intensity."),
        ("Dance Cardio Party", "cardio", "beginner", 45, "weight_loss",
         "Fun dance-inspired cardio session mixing salsa, hip-hop, and aerobic moves for full-body calorie burn."),
        ("Rowing Power Session", "cardio", "advanced", 30, "endurance",
         "Intense rowing intervals targeting cardiovascular capacity and full-body muscular endurance."),
        ("Trail Run Adventure", "cardio", "intermediate", 50, "endurance",
         "Outdoor trail running workout on varied terrain to build stamina, agility, and mental toughness."),
        ("Elliptical Fat Burner", "cardio", "beginner", 35, "weight_loss",
         "Low-impact elliptical session with programmed intervals to burn fat while protecting joints."),
        ("Cardio Boxing Blast", "cardio", "intermediate", 30, "weight_loss",
         "Boxing-inspired cardio workout combining jabs, crosses, hooks, and footwork for total body conditioning."),

        # Strength plans
        ("Full Body Strength", "strength", "intermediate", 45, "muscle_gain",
         "Compound movement workout hitting all major muscle groups with squats, deadlifts, bench press, and rows."),
        ("Upper Body Power", "strength", "advanced", 40, "muscle_gain",
         "Heavy upper body session focusing on progressive overload for chest, shoulders, back, and arms."),
        ("Lower Body Builder", "strength", "intermediate", 50, "muscle_gain",
         "Leg-focused strength training with squats, lunges, leg press, and calf raises for muscle hypertrophy."),
        ("Core Strength Foundation", "strength", "beginner", 25, "muscle_gain",
         "Foundational core workout with planks, crunches, leg raises, and stability exercises for abdominal strength."),
        ("Dumbbell Total Body", "strength", "beginner", 35, "muscle_gain",
         "Accessible dumbbell-only workout covering all muscle groups, perfect for home or gym training."),
        ("Powerlifting Basics", "strength", "advanced", 60, "muscle_gain",
         "Focused powerlifting session on the big three: squat, bench press, and deadlift with heavy singles and triples."),
        ("Kettlebell Strength Flow", "strength", "intermediate", 35, "muscle_gain",
         "Dynamic kettlebell workout combining swings, cleans, presses, and Turkish get-ups for functional strength."),
        ("Bodyweight Strength", "strength", "beginner", 30, "muscle_gain",
         "No-equipment strength routine using push-ups, pull-ups, dips, and squats for progressive calisthenics training."),
        ("Olympic Lifting Intro", "strength", "advanced", 55, "muscle_gain",
         "Introduction to Olympic lifts including clean and jerk and snatch variations for explosive power development."),
        ("Resistance Band Power", "strength", "beginner", 25, "muscle_gain",
         "Resistance band workout providing variable tension for muscle activation and strength building anywhere."),

        # Yoga plans
        ("Sunrise Yoga Flow", "yoga", "beginner", 30, "flexibility",
         "Gentle morning yoga sequence with sun salutations, warrior poses, and seated stretches for flexibility and calm."),
        ("Power Yoga Challenge", "yoga", "advanced", 60, "flexibility",
         "Demanding power yoga session with arm balances, inversions, and deep backbends for advanced practitioners."),
        ("Restorative Yoga", "yoga", "beginner", 45, "flexibility",
         "Deeply relaxing restorative yoga with supported poses, bolsters, and long holds for stress relief and recovery."),
        ("Yoga for Athletes", "yoga", "intermediate", 40, "flexibility",
         "Sport-specific yoga targeting hip openers, hamstring stretches, and shoulder mobility for athletic performance."),
        ("Vinyasa Flow", "yoga", "intermediate", 50, "flexibility",
         "Flowing vinyasa sequence linking breath to movement through creative transitions and challenging holds."),
        ("Yin Yoga Deep Stretch", "yoga", "beginner", 55, "flexibility",
         "Slow-paced yin yoga with passive floor poses held 3-5 minutes to target deep connective tissue and fascia."),
        ("Hot Yoga Intensity", "yoga", "advanced", 60, "flexibility",
         "Bikram-inspired hot yoga series of 26 postures performed in heated room for deep flexibility and detoxification."),
        ("Chair Yoga Basics", "yoga", "beginner", 20, "flexibility",
         "Accessible chair-based yoga routine perfect for office workers, seniors, or those with mobility limitations."),
        ("Ashtanga Primary Series", "yoga", "advanced", 75, "flexibility",
         "Traditional Ashtanga primary series with set sequence of postures building heat, strength, and deep flexibility."),
        ("Yoga Sculpt", "yoga", "intermediate", 45, "muscle_gain",
         "Hybrid yoga session incorporating light weights and high-rep exercises into traditional yoga flow for toning."),

        # HIIT plans
        ("HIIT Inferno", "HIIT", "advanced", 25, "weight_loss",
         "All-out high-intensity interval training with burpees, mountain climbers, and plyometric jumps for maximum calorie burn."),
        ("Tabata Thunder", "HIIT", "advanced", 20, "weight_loss",
         "Classic Tabata protocol: 20 seconds max effort, 10 seconds rest for 8 rounds across multiple exercises."),
        ("Beginner HIIT", "HIIT", "beginner", 20, "weight_loss",
         "Modified high-intensity intervals with longer rest periods and lower impact exercises for HIIT newcomers."),
        ("HIIT and Strength Combo", "HIIT", "intermediate", 35, "muscle_gain",
         "Hybrid workout alternating strength exercises with cardio bursts for simultaneous muscle building and fat loss."),
        ("Metabolic Conditioning", "HIIT", "intermediate", 30, "weight_loss",
         "Metabolic circuit combining compound movements at high intensity to elevate metabolism for hours post-workout."),
        ("EMOM Challenge", "HIIT", "advanced", 30, "endurance",
         "Every-Minute-On-the-Minute workout with escalating rep schemes to push cardiovascular and muscular endurance limits."),
        ("Low Impact HIIT", "HIIT", "beginner", 25, "weight_loss",
         "Joint-friendly HIIT session replacing jumps with low-impact alternatives while maintaining high heart rate."),
        ("HIIT Ladder Workout", "HIIT", "intermediate", 30, "endurance",
         "Ascending and descending rep ladder with bodyweight exercises for sustained intensity and endurance building."),
        ("Spartan HIIT", "HIIT", "advanced", 35, "endurance",
         "Military-inspired HIIT combining bear crawls, burpees, sprints, and carries for extreme functional conditioning."),
        ("HIIT Yoga Fusion", "HIIT", "intermediate", 40, "weight_loss",
         "Unique fusion alternating intense HIIT circuits with yoga recovery flows for balanced fitness and flexibility."),

        # Pilates plans
        ("Classical Pilates Mat", "pilates", "intermediate", 45, "flexibility",
         "Traditional mat Pilates sequence focusing on the powerhouse, spinal articulation, and controlled precise movements."),
        ("Pilates for Core", "pilates", "beginner", 30, "muscle_gain",
         "Core-focused Pilates routine with hundred, roll-up, single leg stretch, and other foundational exercises."),
        ("Advanced Pilates Flow", "pilates", "advanced", 50, "flexibility",
         "Challenging Pilates session with teaser, control balance, and boomerang for experienced practitioners."),
        ("Pilates Reformer Basics", "pilates", "beginner", 40, "flexibility",
         "Introductory reformer Pilates covering footwork, leg circles, and basic arm springs for whole-body conditioning."),
        ("Pilates Barre Blend", "pilates", "intermediate", 45, "muscle_gain",
         "Ballet barre-inspired Pilates workout with small pulsing movements for lean muscle toning and posture improvement."),

        # Stretching plans
        ("Morning Mobility Routine", "stretching", "beginner", 15, "flexibility",
         "Quick morning mobility sequence with dynamic stretches targeting hips, shoulders, and thoracic spine."),
        ("Post-Workout Cool Down", "stretching", "beginner", 15, "flexibility",
         "Essential cool-down stretches for all major muscle groups to improve recovery and reduce soreness."),
        ("Deep Flexibility Program", "stretching", "intermediate", 40, "flexibility",
         "Progressive flexibility training with PNF stretching techniques for splits, backbends, and shoulder flexibility."),
        ("Foam Rolling Recovery", "stretching", "beginner", 20, "flexibility",
         "Self-myofascial release session using foam roller to break up adhesions and improve tissue quality."),
        ("Active Recovery Stretch", "stretching", "intermediate", 30, "flexibility",
         "Light movement and stretching routine designed for rest days to promote blood flow and maintain flexibility."),

        # Mixed / Cross-training
        ("CrossFit WOD", "cross_training", "advanced", 45, "endurance",
         "CrossFit-style workout of the day combining weightlifting, gymnastics, and cardio for high-intensity functional fitness."),
        ("Functional Fitness Circuit", "cross_training", "intermediate", 40, "endurance",
         "Circuit-based functional training with battle ropes, box jumps, sled pushes, and farmer carries."),
        ("Boot Camp Blast", "cross_training", "intermediate", 45, "weight_loss",
         "Military boot camp-inspired outdoor circuit with running, calisthenics, and partner exercises."),
        ("TRX Suspension Training", "cross_training", "intermediate", 35, "muscle_gain",
         "Full-body TRX suspension workout using bodyweight and gravity for adjustable resistance training."),
        ("Obstacle Course Prep", "cross_training", "advanced", 50, "endurance",
         "Obstacle course race preparation combining running, climbing, crawling, and grip strength exercises."),
        ("Swimming Endurance", "cross_training", "intermediate", 45, "endurance",
         "Pool-based endurance workout with lap swimming, drills, and interval sets for cardiovascular and muscular endurance."),
        ("Martial Arts Conditioning", "cross_training", "advanced", 40, "endurance",
         "Martial arts-inspired conditioning with kicks, punches, agility drills, and core work for combat readiness."),
        ("Senior Fitness Program", "cross_training", "beginner", 30, "flexibility",
         "Safe, low-impact fitness program for seniors combining gentle strength work, balance exercises, and stretching."),
        ("Prenatal Fitness", "cross_training", "beginner", 30, "flexibility",
         "Pregnancy-safe workout with modified strength, pelvic floor exercises, and gentle stretching for maternal fitness."),
        ("Weekend Warrior Circuit", "cross_training", "intermediate", 50, "weight_loss",
         "Challenging weekend workout combining elements of strength, cardio, and agility for comprehensive fitness."),
    ]

    plans = pd.DataFrame(plan_templates,
                         columns=["name", "type", "difficulty", "duration_min", "target_goal", "description"])
    plans.insert(0, "plan_id", range(1, len(plans) + 1))
    return plans


# ============================================================
# 3. Generate Ratings (Realistic Simulation)
# ============================================================
def generate_ratings(users, plans, min_ratings=MIN_RATINGS):
    """
    Generate realistic user-plan ratings.
    Users are more likely to rate — and rate higher — plans that match their
    goal and fitness level.
    """
    ratings_list = []

    difficulty_map = {"beginner": 0, "intermediate": 1, "advanced": 2}
    fitness_map = {"beginner": 0, "intermediate": 1, "advanced": 2}

    for _, user in users.iterrows():
        # Each user rates between 5 and 15 plans
        n_ratings = np.random.randint(5, 16)
        # Bias plan selection towards matching goal
        plan_weights = np.ones(len(plans))
        goal_match = plans["target_goal"] == user["goal"]
        plan_weights[goal_match] = 3.0  # 3x more likely to rate matching plans
        plan_weights /= plan_weights.sum()

        rated_plan_ids = np.random.choice(plans["plan_id"], size=min(n_ratings, len(plans)),
                                          replace=False, p=plan_weights)

        for pid in rated_plan_ids:
            plan = plans[plans["plan_id"] == pid].iloc[0]

            # Base rating
            base = 3.0

            # Bonus for goal match
            if plan["target_goal"] == user["goal"]:
                base += 1.0

            # Bonus/penalty for difficulty match
            user_level = fitness_map.get(user["fitness_level"], 1)
            plan_level = difficulty_map.get(plan["difficulty"], 1)
            diff = abs(user_level - plan_level)
            if diff == 0:
                base += 0.5
            elif diff == 2:
                base -= 1.0

            # Add noise
            rating = base + np.random.normal(0, 0.6)
            rating = int(np.clip(np.round(rating), 1, 5))

            ratings_list.append({
                "user_id": user["user_id"],
                "plan_id": int(pid),
                "rating": rating
            })

    ratings = pd.DataFrame(ratings_list)

    # Ensure minimum ratings count by adding more if needed
    while len(ratings) < min_ratings:
        extra_user = np.random.randint(1, NUM_USERS + 1)
        extra_plan = np.random.randint(1, NUM_PLANS + 1)
        # Avoid duplicates
        if not ((ratings["user_id"] == extra_user) & (ratings["plan_id"] == extra_plan)).any():
            ratings = pd.concat([ratings, pd.DataFrame([{
                "user_id": extra_user,
                "plan_id": extra_plan,
                "rating": np.random.randint(1, 6)
            }])], ignore_index=True)

    return ratings


# ============================================================
# Main — Generate and save datasets
# ============================================================
def main():
    print("=" * 60)
    print("FitMatch - Dataset Generation")
    print("=" * 60)

    users = generate_users()
    plans = generate_workout_plans()
    ratings = generate_ratings(users, plans)

    # Save
    users.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
    plans.to_csv(os.path.join(OUTPUT_DIR, "workout_plans.csv"), index=False)
    ratings.to_csv(os.path.join(OUTPUT_DIR, "ratings.csv"), index=False)

    # Summary
    total_possible = NUM_USERS * NUM_PLANS
    sparsity = 1 - len(ratings) / total_possible

    print(f"\n✅ Users generated:          {len(users)}")
    print(f"✅ Workout plans generated:  {len(plans)}")
    print(f"✅ Ratings generated:        {len(ratings)}")
    print(f"📊 Sparsity:                 {sparsity:.2%}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
