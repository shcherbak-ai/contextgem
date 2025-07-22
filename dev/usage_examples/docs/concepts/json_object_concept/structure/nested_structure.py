from contextgem import JsonObjectConcept


device_config_concept = JsonObjectConcept(
    name="Device Configuration",
    description="Configuration details for a networked device",
    structure={
        "device": {"id": str, "type": str, "model": str},
        "network": {"ip_address": str, "subnet_mask": str, "gateway": str},
        "settings": {"enabled": bool, "mode": str},
    },
)
