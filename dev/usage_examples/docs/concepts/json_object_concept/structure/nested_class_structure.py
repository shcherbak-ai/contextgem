from dataclasses import dataclass

from contextgem import JsonObjectConcept


# Use dataclasses to define the structure of the JSON object
# (Pydantic models and other classes with type annotations work the same way)


@dataclass
class Location:
    latitude: float
    longitude: float
    altitude: float


@dataclass
class Sensor:
    id: str
    type: str
    location: Location  # reference to another class
    active: bool


@dataclass
class SensorNetwork:
    network_id: str
    primary_sensor: Sensor  # reference to another class
    backup_sensors: list[Sensor]  # list of another class


sensor_network_concept = JsonObjectConcept(
    name="IoT Sensor Network",
    description="Configuration for a network of IoT sensors",
    structure=SensorNetwork,  # nested class structure
)
