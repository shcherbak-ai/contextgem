from dataclasses import dataclass

from contextgem import JsonObjectClassStruct, JsonObjectConcept


@dataclass
class Address(JsonObjectClassStruct):
    street: str
    city: str
    country: str


@dataclass
class Contact(JsonObjectClassStruct):
    email: str
    phone: str
    address: Address


@dataclass
class Person(JsonObjectClassStruct):
    name: str
    age: int
    contact: Contact


# Use the class structure with JsonObjectConcept
# JsonObjectClassStruct enables automatic conversion of typed class hierarchies
# into the dictionary structure required by JsonObjectConcept, preserving the
# type information and nested relationships between classes.
JsonObjectConcept(name="person", description="Person information", structure=Person)
