from __future__ import annotations

from dataclasses import dataclass

from contextgem import JsonObjectClassStruct, JsonObjectConcept


def create_person_concept() -> JsonObjectConcept:
    # Classes defined inside a function, in a module with postponed annotations
    # (`from __future__ import annotations`), where the type of `Contact.address`
    # is the string "Address". As they inherit from JsonObjectClassStruct, "Address"
    # is found among the classes of the function. Classes defined at module level
    # need no base class.
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

    return JsonObjectConcept(
        name="Person", description="Person information", structure=Person
    )


person_concept = create_person_concept()
