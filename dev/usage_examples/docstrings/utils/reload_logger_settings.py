import os

from contextgem import reload_logger_settings


# Initial logger settings are loaded from environment variables at import time

# Change logger level to WARNING
os.environ["CONTEXTGEM_LOGGER_LEVEL"] = "WARNING"
print("Setting logger level to WARNING")
reload_logger_settings()
# Now the logger will only show WARNING level and above messages

# Disable the logger completely
os.environ["CONTEXTGEM_LOGGER_LEVEL"] = "OFF"
print("Disabling the logger")
reload_logger_settings()
# Now the logger is disabled and won't show any messages

# You can re-enable the logger by setting it back to a valid level
# os.environ["CONTEXTGEM_LOGGER_LEVEL"] = "INFO"
# reload_logger_settings()
