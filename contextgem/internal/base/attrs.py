#
# ContextGem
#
# Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Module defining base classes for attribute processing in the ContextGem framework.

This module provides foundational classes that handle various attribute types including:
- Assigned instances (aspects and concepts)
- Property validation and processing
- Extracted items management
- Reference paragraphs and sentences

These base classes implement common functionality for attribute validation,
manipulation, and access control that is inherited by higher-level components.
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from pydantic import Field, StrictBool, StrictInt

if TYPE_CHECKING:
    from contextgem.public.aspects import Aspect
    from contextgem.internal.base.concepts import _Concept
    from contextgem.internal.items import _ExtractedItem

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.decorators import _post_init_method
from contextgem.internal.typings.aliases import (
    AssignedInstancesAttrName,
    JustificationDepth,
    Self,
)
from contextgem.public.paragraphs import Paragraph
from contextgem.public.sentences import Sentence


class _AssignedInstancesAttributeProcessor(_InstanceBase):
    """
    Handles assigned instances-related attributes of a subclass' instance.
    """

    @property
    @abstractmethod
    def llm_roles(self) -> set[str]:
        """
        Abstract property, to be implemented by subclasses.

        Returns a set of LLM roles associated with the object's assigned instances.

        :return: set
        """
        pass

    def _add_instances(
        self,
        attr_name: AssignedInstancesAttrName,
        instances: list[Aspect | _Concept],
    ) -> Self:
        """
        Adds a list of new instances to an existing attribute's collection. This
        method ensures existing instances are retained while appending a deep copy
        of the provided instances to the specified attribute.

        :param attr_name: Name of the attribute which holds the instances.
        :type attr_name: AssignedInstancesAttrName
        :param instances: List of instances to be added to the attribute.
        :type instances: list[Aspect | _Concept]
        :return: Updated object for method chaining.
        :rtype: Self
        """
        current_instances = getattr(self, attr_name)
        new_instances = deepcopy(instances)
        combined_instances = current_instances + new_instances
        # Use a temporary instance to test if the combined list would be valid
        # without modifying the current instance
        setattr(self.clone(), attr_name, combined_instances)
        # If we get here, validation passed, so update the actual instance
        setattr(self, attr_name, combined_instances)
        return self

    def _get_instance_by_name(
        self, attr_name: AssignedInstancesAttrName, instance_name: str
    ) -> Aspect | _Concept:
        """
        Retrieves and returns a specific instance from a collection
        of instances managed by an attribute of the class.

        :param attr_name: Name of the attribute that holds the collection of instances.
        :type attr_name: AssignedInstancesAttrName
        :param instance_name: Name of the instance to retrieve from the list.
        :type instance_name: str
        :return: The instance with the specified name if found.
        :rtype: Aspect | _Concept
        :raises ValueError: If no instance with the specified name is found.
        """
        instances = getattr(self, attr_name)
        try:
            return next(i for i in instances if i.name == instance_name)
        except StopIteration:
            raise ValueError(
                f"{attr_name.title()[:-1]} with name {instance_name} not found."
            )

    def _remove_instance_by_name(
        self, attr_name: AssignedInstancesAttrName, instance_name: str
    ) -> Self:
        """
        Removes an instance from an attribute containing a list of instances by its name.

        :param attr_name: The name of the attribute containing the list of
            instances to operate on.
        :type attr_name: AssignedInstancesAttrName
        :param instance_name: The name of the instance to be removed from the
            list of instances.
        :type instance_name: str

        :return: The object itself after the instance has been removed,
            enabling method chaining.
        :rtype: Self
        """
        instances = getattr(self, attr_name)
        self._get_instance_by_name(
            attr_name=attr_name, instance_name=instance_name
        )  # Validate item exists
        setattr(self, attr_name, [i for i in instances if i.name != instance_name])
        return self

    def _remove_all_instances(
        self,
        attr_name: AssignedInstancesAttrName,
    ) -> Self:
        """
        Removes all instances from the specified attribute and returns
        the modified object.

        :param attr_name: Specifies the name of the attribute where all instances
            need to be removed. The attribute identified by this name must
            be a list.
        :return: The updated object with the specified attribute cleared.
        """
        setattr(self, attr_name, [])
        return self


class _AssignedAspectsProcessor(_AssignedInstancesAttributeProcessor):
    """
    Base class to be inherited by subclasses with assigned aspects.
    """

    @_post_init_method
    def _post_init(self, __context):
        if not hasattr(self, "aspects"):
            raise AttributeError("Instance has no `aspects` attribute.")

    @property
    def llm_roles(self) -> set[str]:
        """
        A set of LLM roles associated with the object's aspects and aspects' concepts.

        :return: A set containing unique LLM roles gathered from aspects and aspects' concepts.
        :rtype: set[str]
        """
        llm_roles = set()
        for aspect in self.aspects:
            llm_roles.add(aspect.llm_role)
            for concept in aspect.concepts:
                llm_roles.add(concept.llm_role)
        return llm_roles

    def add_aspects(
        self,
        aspects: list[Aspect],
    ) -> Self:
        """
        Adds aspects to the existing aspects list of an instance and returns the
        updated instance. This method ensures that the provided aspects are deeply
        copied to avoid any unintended state modification of the original reusable
        aspects.

        :param aspects: A list of aspects to be added. Each aspect is deeply copied
            to ensure the original list remains unaltered.
        :type aspects: list[Aspect]

        :return: Updated instance containing the newly added aspects.
        :rtype: Self
        """
        return self._add_instances(attr_name="aspects", instances=aspects)

    def get_aspect_by_name(self, name: str) -> Aspect:
        """
        Finds and returns an aspect with the specified name from the list of available aspects,
        if the instance has `aspects` attribute.

        :param name: The name of the aspect to find.
        :type name: str
        :return: The aspect with the specified name.
        :rtype: Aspect
        :raises ValueError: If no aspect with the specified name is found.
        """
        return self._get_instance_by_name(attr_name="aspects", instance_name=name)

    def get_aspects_by_names(self, names: list[str]) -> list[Aspect]:
        """
        Retrieve a list of Aspect objects corresponding to the provided list of names.

        :param names: List of aspect names to retrieve. The names must be provided
                      as a list of strings.
        :returns: A list of Aspect objects corresponding to provided names.
        :rtype: list[Aspect]
        """
        return [self.get_aspect_by_name(name) for name in names]

    def remove_aspect_by_name(self, name: str) -> Self:
        """
        Removes an aspect from the assigned aspects by its name.

        :param name: The name of the aspect to be removed
        :type name: str
        :return: Updated instance with the aspect removed.
        :rtype: Self
        """
        return self._remove_instance_by_name(attr_name="aspects", instance_name=name)

    def remove_aspects_by_names(self, names: list[str]) -> Self:
        """
        Removes multiple aspects from an object based on the provided list of names.

        :param names: A list of names identifying the aspects to be removed.
        :type names: list[str]
        :return: The updated object after the specified aspects have been removed.
        :rtype: Self
        """
        for name in names:
            self.remove_aspect_by_name(name)
        return self

    def remove_all_aspects(self) -> Self:
        """
        Removes all aspects from the instance and returns the updated instance.

        This method clears the `aspects` attribute of the instance by resetting it to
        an empty list. It returns the same instance, allowing for method chaining.

        :return: The updated instance with all aspects removed
        """
        return self._remove_all_instances(attr_name="aspects")


class _AssignedConceptsProcessor(_AssignedInstancesAttributeProcessor):
    """
    Base class to be inherited by subclasses with assigned concepts.
    """

    @_post_init_method
    def _post_init(self, __context):
        if not hasattr(self, "concepts"):
            raise AttributeError("Instance has no `concepts` attribute.")

    @property
    def llm_roles(self) -> set[str]:
        """
        A set of LLM roles associated with the object's concepts.

        :return: A set containing unique LLM roles gathered from concepts.
        :rtype: set[str]
        """
        llm_roles = set()
        for concept in self.concepts:
            llm_roles.add(concept.llm_role)
        return llm_roles

    def add_concepts(
        self,
        concepts: list[_Concept],
    ) -> Self:
        """
        Adds a list of new concepts to the existing `concepts` attribute of the instance. This method ensures
        that the provided list of concepts is deep-copied to prevent unintended side effects from modifying
        the input list outside of this method.

        :param concepts: A list of concepts to be added. It will be deep-copied before being added
            to the instance's `concepts` attribute.
        :type concepts: list[_Concept]
        :return: Returns the instance itself after the modification.
        :rtype: Self
        """
        return self._add_instances(attr_name="concepts", instances=concepts)

    def get_concept_by_name(self, name: str) -> _Concept:
        """
        Retrieves a concept from the list of concepts based on the provided name,
        if the instance has `concepts` attribute.

        :param name: The name of the concept to search for.
        :type name: str
        :return: The `_Concept` object with the specified name.
        :rtype: _Concept
        :raises ValueError: If no concept with the specified name is found.
        """
        return self._get_instance_by_name(attr_name="concepts", instance_name=name)

    def get_concepts_by_names(self, names: list[str]) -> list[_Concept]:
        """
        Retrieve a list of _Concept objects corresponding to the provided list of names.

        :param names: List of concept names to retrieve. The names must be provided
                      as a list of strings.
        :returns: A list of _Concept objects corresponding to provided names.
        :rtype: list[_Concept]
        """
        return [self.get_concept_by_name(name) for name in names]

    def remove_concept_by_name(self, name: str) -> Self:
        """
        Removes a concept from the assigned concepts by its name.

        :param name: The name of the concept to be removed
        :type name: str
        :return: Updated instance with the concept removed.
        :rtype: Self
        """
        return self._remove_instance_by_name(attr_name="concepts", instance_name=name)

    def remove_concepts_by_names(self, names: list[str]) -> Self:
        """
        Removes concepts from the object by their names.

        :param names: A list of concept names to be removed.
        :type names: list[str]
        :return: Returns the updated instance after removing the specified concepts.
        :rtype: Self
        """
        for name in names:
            self.remove_concept_by_name(name)
        return self

    def remove_all_concepts(self) -> Self:
        """
        Removes all concepts from the instance and returns the updated instance.

        This method clears the `concepts` attribute of the instance by resetting it to
        an empty list. It returns the same instance, allowing for method chaining.

        :return: The updated instance with all concepts removed
        """
        return self._remove_all_instances(attr_name="concepts")


class _AssignedInstancesProcessor(
    _AssignedAspectsProcessor, _AssignedConceptsProcessor
):
    """
    Base class to be inherited by subclasses with all assigned instance types.
    """

    @property
    def llm_roles(self) -> set[str]:
        """
        A set of LLM roles associated with the object's aspects and concepts.

        :return: A set containing unique LLM roles gathered from aspects and concepts.
        :rtype: set[str]
        """
        aspects_roles = _AssignedAspectsProcessor.llm_roles.fget(self)
        concepts_roles = _AssignedConceptsProcessor.llm_roles.fget(self)
        return aspects_roles | concepts_roles

    def remove_all_instances(self) -> Self:
        """
        Removes all assigned instances from the object and resets
        them as empty lists. Returns the modified instance.

        :return: The modified object with all assigned instances removed.
        :rtype: Self
        """
        return self.remove_all_aspects().remove_all_concepts()


class _PropertyProcessor(_InstanceBase):
    """
    Base class for processing and managing properties.
    """

    def _validate_and_assign_list_property(
        self, value: list, expected_type: type, attr_name: str, error_message: str
    ) -> None:
        """
        Validates and assigns a list property to an object.

        Ensures that the provided `value` is a list where all elements are instances of
        the specified `expected_type` and their IDs are unique. If valid, assigns the
        list to the specified object attribute. Otherwise, raises a `ValueError` with
        the given error message.

        :param value: The value to validate and assign. Must be a list of elements of
            type `expected_type`.
        :param expected_type: The expected type of elements in the list.
        :param attr_name: The name of the attribute to assign the value to.
        :param error_message: The error message to raise in case of a validation error.
        :return: None.
        """
        if not isinstance(value, list) or not all(
            isinstance(item, expected_type) for item in value
        ):
            raise ValueError(error_message)
        self._validate_list_uniqueness(value)
        setattr(self, attr_name, value)


class _ExtractedItemsAttributeProcessor(_PropertyProcessor):
    """
    Handles the processing of extracted items and validates their assignment
    and retrieval.

    :ivar add_justifications: A boolean flag indicating whether the LLM
        will output justification for each extracted item for the aspect. Defaults to False.
    :vartype add_justifications: bool
    :ivar justification_depth: The level of detail of justifications. Details to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: The maximum number of sentences in a justification.
        Defaults to 2.
    :vartype justification_max_sents: int
    """

    add_justifications: StrictBool = Field(default=False)
    justification_depth: JustificationDepth = Field(default="brief")
    justification_max_sents: StrictInt = Field(default=2)

    @_post_init_method
    def _post_init(self, __context):
        if not hasattr(self, "_extracted_items"):
            raise AttributeError("Instance has no `_extracted_items` attribute.")

    @property
    @abstractmethod
    def _item_class(self) -> type[_ExtractedItem]:
        """
        Abstract property, to be implemented by subclasses.

        Returns the specific extracted class type for the instance.

        :rtype: type[_ExtractedItem]
        :return: The extracted class type for the instance.
        """
        pass

    @property
    def extracted_items(self) -> list[_ExtractedItem]:
        """
        Provides access to extracted items.

        :return: A list containing the extracted items as `_ExtractedItem` objects.
        :rtype: list[_ExtractedItem]
        """
        return self._extracted_items

    @extracted_items.setter
    def extracted_items(self, value: list[_ExtractedItem]) -> None:
        """
        Sets the extracted_items property of the object. Ensures that the value provided
        is a list where all elements are instances of the relevant _ExtractedItem subclass
        and their IDs are unique.

        :param value: The new list of extracted items to be set.
        :type value: list[_ExtractedItem]
        :raises ValueError: If the value is not a list or if any elements in the list
            are not instances of the relevant _ExtractedItem subclass.
        :return: None
        """
        self._validate_and_assign_list_property(
            value,
            self._item_class,
            "_extracted_items",
            f"Extracted items must be a list of `{self._item_class.__name__}` objects",
        )


class _RefParasAndSentsAttrituteProcessor(_PropertyProcessor):
    """
    Base class that handles processing and validation of reference paragraphs and reference sentences attributes.
    """

    @_post_init_method
    def _post_init(self, __context):
        for attr_name in ("_reference_paragraphs", "_reference_sentences"):
            if not hasattr(self, attr_name):
                raise AttributeError(f"Instance has no `{attr_name}` attribute.")

    @property
    def reference_paragraphs(self) -> list[Paragraph]:
        """
        Provides access to the instance's reference paragraphs, assigned during extraction.

        :return: A list containing the paragraphs as `Paragraph` objects.
        :rtype: list[Paragraph]
        """
        return self._reference_paragraphs

    @reference_paragraphs.setter
    def reference_paragraphs(self, value: list[Paragraph]) -> None:
        """
        Sets the `reference_paragraphs` property of the instance.

        :param value: The new list of paragraphs to be set.
        :type value: list[Paragraph]
        :raises ValueError: If the value is not a list or if any elements in the list
            are not instances of the Paragraph class.
        :return: None
        """
        self._validate_and_assign_list_property(
            value,
            Paragraph,
            "_reference_paragraphs",
            "Paragraphs must be a list of `Paragraph` objects",
        )

    @property
    def reference_sentences(self) -> list[Sentence]:
        """
        Provides access to the instance's reference sentences, assigned during extraction.

        :return: A list containing the sentences as `Sentence` objects.
        :rtype: list[Sentence]
        """
        return self._reference_sentences

    @reference_sentences.setter
    def reference_sentences(self, value: list[Sentence]) -> None:
        """
        Sets the `reference_sentences` property of the instance.

        :param value: The new list of sentences to be set.
        :type value: list[Sentence]
        :raises ValueError: If the value is not a list or if any elements in the list
            are not instances of the Sentence class.
        :return: None
        """
        self._validate_and_assign_list_property(
            value,
            Sentence,
            "_reference_sentences",
            "Sentences must be a list of `Sentence` objects",
        )
