from contextgem import RatingScale

# Create a rating scale with default values (0 to 10)
default_scale = RatingScale()

# Create a custom rating scale (1 to 5)
custom_scale = RatingScale(
    start=1,
    end=5,
)

# RatingScale objects are immutable
try:
    custom_scale.end = 7
except ValueError as e:
    print(f"Error when trying to modify rating scale: {e}")
