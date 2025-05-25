from dataclasses import dataclass

from contextgem import JsonObjectConcept
from contextgem.public.utils import JsonObjectClassStruct

# Use dataclasses to define the structure of the JSON object


# All classes in the nested class structure must inherit from JsonObjectClassStruct
# to enable automatic conversion of the class hierarchy to a dictionary structure
# for JsonObjectConcept
@dataclass
class Location(JsonObjectClassStruct):
    latitude: float
    longitude: float
    altitude: float


@dataclass
class Sensor(JsonObjectClassStruct):
    id: str
    type: str
    location: Location  # reference to another class
    active: bool


@dataclass
class SensorNetwork(JsonObjectClassStruct):
    network_id: str
    primary_sensor: Sensor  # reference to another class
    backup_sensors: list[Sensor]  # list of another class


sensor_network_concept = JsonObjectConcept(
    name="IoT Sensor Network",
    description="Configuration for a network of IoT sensors",
    structure=SensorNetwork,  # nested class structure
)
