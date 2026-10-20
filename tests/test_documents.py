# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for documents and text segmentation.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import shutil
import subprocess  # nosec B404 - runs the Python of the tests, no shell
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from huggingface_hub import try_to_load_from_cache

from contextgem.internal import utils as utils_module
from contextgem.internal.base import documents as documents_module
from contextgem.internal.utils import (
    _SAT_BATCH_SIZE,
    _SAT_MODELS_IN_USE,
    _get_sat_model,
    _load_sat_model,
    _split_into_sentences,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    Document,
    ExtractionPipeline,
    Paragraph,
    Sentence,
    StringConcept,
    release_sat_model,
)
from tests.base import BaseTest
from tests.utils import get_test_document_text


class _StubSaTModel:
    """
    Stands in for a SaT model: splits a text into paragraphs at blank lines,
    and a paragraph into sentences after each full stop.
    """

    def split(
        self,
        text_or_texts: str | list[str],
        do_paragraph_segmentation: bool = False,
        **kwargs: Any,
    ) -> list[list[str]]:
        """
        Splits a text into paragraphs, each given as its sentences, or each of
        the texts of paragraphs into sentences.

        :param text_or_texts: The text, or the texts of the paragraphs.
        :type text_or_texts: str | list[str]
        :param do_paragraph_segmentation: Whether to split a text into paragraphs.
        :type do_paragraph_segmentation: bool
        :param kwargs: The other options of the segmentation, e.g. the batch
            size, which the stub does not use.
        :type kwargs: Any
        :return: The sentences of each paragraph.
        :rtype: list[list[str]]
        """
        if do_paragraph_segmentation:
            # Each paragraph is given as one sentence
            return [[paragraph] for paragraph in str(text_or_texts).split("\n\n")]
        return [re.split(r"(?<=\.) ", text) for text in text_or_texts]


@pytest.fixture
def release_kept_sat_model() -> Generator[None]:
    """
    Releases the SaT model that is kept loaded before and after the test, so that
    the test starts with no model kept, and that the tests that follow do not get
    a model that it kept, e.g. a stub.
    """
    release_sat_model()
    yield
    release_sat_model()


class TestDocuments(BaseTest):
    """
    Tests for documents, paragraphs, sentences and extraction pipelines,
    and for the segmentation of text with SaT models.
    """

    # Shared objects passed to the `parametrize` decorators of the tests below.
    # The decorators are evaluated in the class body, where the attributes inherited
    # from the base class are not in scope, so the names are bound here.
    document = BaseTest.document
    extraction_pipeline = BaseTest.extraction_pipeline

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_kept_until_released(self):
        """
        Tests that the SaT model used last is kept loaded, and is released by
        `release_sat_model()` as soon as nothing else holds it.
        """
        model = _get_sat_model("sat-3l-sm")
        model_ref = weakref.ref(model)
        del model
        # Kept, whatever form its ID is given in
        assert model_ref() is not None
        assert _get_sat_model("sat-3l-sm") is model_ref()
        assert _get_sat_model(Path("sat-3l-sm")) is model_ref()
        release_sat_model()
        # Released without a garbage collection, i.e. as the last reference to it
        # is dropped: the model is in no reference cycle
        assert model_ref() is None
        assert "sat-3l-sm" not in _SAT_MODELS_IN_USE

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_loaded_once_for_threads(self):
        """
        Tests that threads that need a SaT model at the same time load it once,
        and share it.
        """
        n_threads = 8
        loaded: list[_StubSaTModel] = []

        def load_slowly(model_id: str) -> _StubSaTModel:
            """
            Loads a model slowly, so that the threads ask for it while it is loaded.
            """
            time.sleep(0.1)
            loaded.append(_StubSaTModel())
            return loaded[-1]

        started = threading.Barrier(n_threads)
        done = threading.Barrier(n_threads)
        models: list[Any] = [None] * n_threads

        def get_model(position: int) -> None:
            """
            Gets the model, and holds it until each thread has got it.
            """
            started.wait()
            models[position] = _get_sat_model("stub-model")
            done.wait()

        with patch.object(utils_module, "_load_sat_model", new=load_slowly):
            threads = [
                threading.Thread(target=get_model, args=(i,)) for i in range(n_threads)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert len(loaded) == 1
            assert all(model is loaded[0] for model in models)

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_loading_does_not_block_other_models(self):
        """
        Tests that the threads that need other SaT models than the one that is
        being loaded, e.g. downloaded, do not wait for it.
        """
        loading_started = threading.Event()
        loading_released = threading.Event()

        def load(model_id: str) -> _StubSaTModel:
            """
            Loads a model, the slow one only when the test releases it.
            """
            if model_id == "slow-model":
                loading_started.set()
                loading_released.wait(timeout=30)
            return _StubSaTModel()

        slow_models: list[Any] = []
        other_models: list[Any] = []

        def get_other_models() -> None:
            """
            Gets the model in use, and another model that is loaded.
            """
            other_models.append(_get_sat_model("model-in-use"))
            other_models.append(_get_sat_model("other-model"))

        with patch.object(utils_module, "_load_sat_model", new=load):
            model_in_use = _get_sat_model("model-in-use")
            slow_loading = threading.Thread(
                target=lambda: slow_models.append(_get_sat_model("slow-model"))
            )
            slow_loading.start()
            try:
                assert loading_started.wait(timeout=30)
                other_loading = threading.Thread(target=get_other_models)
                other_loading.start()
                other_loading.join(timeout=10)
                assert not other_loading.is_alive(), "waited for the slow model"
            finally:
                loading_released.set()
                slow_loading.join()
            assert other_models[0] is model_in_use
            assert len(other_models) == 2
            assert len(slow_models) == 1

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_shared_by_paths_of_same_folder(self, tmp_path, monkeypatch):
        """
        Tests that the paths of a folder with a SaT model are those of one model,
        which is loaded once, in whatever form they are given, and that the same
        relative path is that of another model once the current directory has
        changed.
        """
        # The models loaded, recorded without holding them
        loaded: list[weakref.ref[_StubSaTModel]] = []

        def load(model_id: str) -> _StubSaTModel:
            """
            Loads a model.
            """
            model = _StubSaTModel()
            loaded.append(weakref.ref(model))
            return model

        (tmp_path / "models" / "sat").mkdir(parents=True)
        (tmp_path / "other" / "models" / "sat").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        absolute_path = str(tmp_path / "models" / "sat")
        paths = [
            "models/sat",
            "models/sat/",
            os.path.join(".", "models", "..", "models", "sat"),
            absolute_path,
            absolute_path + os.sep,
        ]
        if sys.platform == "win32":
            # Windows paths do not tell the case of the letters apart
            paths.append(absolute_path.upper())
        text = "First sentence. Second sentence."
        with patch.object(utils_module, "_load_sat_model", new=load):
            for sat_model_id in [*paths, Path("models") / "sat"]:
                Document(
                    raw_text=text,
                    sat_model_id=sat_model_id,
                    pre_segment_sentences=True,
                )
            assert len(loaded) == 1
            # The same relative path in another directory: another folder
            monkeypatch.chdir(tmp_path / "other")
            Document(
                raw_text=text,
                sat_model_id="models/sat",
                pre_segment_sentences=True,
            )
            assert len(loaded) == 2

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_kept_for_documents_one_after_another(self):
        """
        Tests that documents processed one after another use the SaT model loaded
        for the first of them, that a model used later is kept in place of
        the one kept before, which is released, and that a document loads
        the model again after `release_sat_model()`.
        """
        # The models loaded, recorded without holding them
        loaded: list[weakref.ref[_StubSaTModel]] = []

        def load(model_id: str) -> _StubSaTModel:
            """
            Loads a model.
            """
            model = _StubSaTModel()
            loaded.append(weakref.ref(model))
            return model

        text = "First sentence. Second sentence.\n\nThird sentence."
        with patch.object(utils_module, "_load_sat_model", new=load):
            for _ in range(3):
                document = Document(raw_text=text, pre_segment_sentences=True)
                assert [len(p.sentences) for p in document.paragraphs] == [2, 1]
            # A paragraph segmentation uses the same model
            Document(raw_text=text, paragraph_segmentation_mode="sat")
            assert len(loaded) == 1
            # Another model is kept in place of the first one, which is released
            Document(
                raw_text=text,
                sat_model_id="sat-12l-sm",
                pre_segment_sentences=True,
            )
            assert len(loaded) == 2
            assert loaded[0]() is None
            assert loaded[1]() is not None
            Document(raw_text=text, pre_segment_sentences=True)
            assert len(loaded) == 3
            assert loaded[1]() is None
            # Released: the next document loads the model again
            release_sat_model()
            assert loaded[2]() is None
            Document(raw_text=text, pre_segment_sentences=True)
            assert len(loaded) == 4

    def test_sentences_segmented_by_tasks_at_the_same_time(self):
        """
        Tests that when two tasks segment the sentences of the same document at
        the same time, each paragraph keeps the sentences of the task that finishes
        first, rather than the sentences of the other task, or a mix of both.
        """
        calls: list[int] = []
        calls_lock = threading.Lock()
        sentences_assigned = threading.Event()

        class SplitterOfTwoTasks:
            """
            Replaces the SaT model: splits a text after each full stop for
            the first task, and into one sentence for the second task, which
            finishes first: the first one waits for its sentences to be assigned.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, which the stub
                    does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                with calls_lock:
                    call = len(calls)
                    calls.append(call)
                if call == 0:
                    assert sentences_assigned.wait(timeout=10)
                    return [re.split(r"(?<=\.) ", text) for text in texts]
                return [[text] for text in texts]

        assign_segmented_sents = Document._assign_segmented_sents

        def assign_and_signal(
            document: Document, split_sents_for_paras: list[list[str]]
        ) -> None:
            """
            Assigns the sentences, and signals that they are assigned.
            """
            assign_segmented_sents(document, split_sents_for_paras)
            sentences_assigned.set()

        document = Document(
            raw_text="First sentence. Second sentence.\n\nThird sentence. Fourth."
        )

        async def segment_twice() -> None:
            """
            Segments the sentences of the document in two tasks at the same time.
            """
            await asyncio.gather(
                document._segment_sents_async(), document._segment_sents_async()
            )

        with (
            patch.object(
                documents_module,
                "_get_sat_model",
                new=lambda model_id: SplitterOfTwoTasks(),
            ),
            patch.object(Document, "_assign_segmented_sents", assign_and_signal),
        ):
            asyncio.run(segment_twice())
        assert calls == [0, 1]
        # The sentences of the second task, which finished first
        assert [[s.raw_text for s in p.sentences] for p in document.paragraphs] == [
            ["First sentence. Second sentence."],
            ["Third sentence. Fourth."],
        ]

    def test_split_into_sentences_in_batches_by_length(self):
        """
        Tests that the texts are split into sentences sorted by length, in batches
        of their own, and that the sentences are returned in the order of the texts.
        """
        calls: list[tuple[list[str], dict[str, Any]]] = []

        class RecordingModel:
            """
            Stands in for a SaT model: records each call, and splits a text into
            its words, so that each text gets sentences of its own.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts.

                :param texts: The texts.
                :type texts: list[str]
                :param kwargs: The options of the segmentation.
                :type kwargs: Any
                :return: The words of each text.
                :rtype: list[list[str]]
                """
                calls.append((texts, kwargs))
                return [text.split() for text in texts]

        texts = ["a b c d", "a", "a b c d e f", "a b", "a b c"]
        model = RecordingModel()
        assert _split_into_sentences(model, texts) == [t.split() for t in texts]  # ty: ignore[invalid-argument-type]
        assert calls == [
            (
                sorted(texts, key=len),
                {
                    "batch_size": _SAT_BATCH_SIZE,
                    "outer_batch_size": _SAT_BATCH_SIZE,
                },
            )
        ]
        assert _split_into_sentences(model, []) == []  # ty: ignore[invalid-argument-type]

    def test_split_into_sentences_as_in_text_order(self):
        """
        Tests that a SaT model splits the texts into the same sentences when they
        are passed to it sorted by length, in batches of their own, as when they
        are passed to it in their order, for short texts and for a text longer than
        the block size of the model, which is split into chunks.
        """
        texts = [
            p.raw_text for p in Document(raw_text=get_test_document_text()).paragraphs
        ]
        # Longer than the 512 tokens of a block of the model
        texts.insert(len(texts) // 2, " ".join(texts))
        model = _get_sat_model("sat-3l-sm")
        assert _split_into_sentences(model, texts) == list(model.split(texts))

    def test_local_sat_model(self):
        """
        Tests the loading of a local SAT model.
        """

        # Test nonexistent path
        non_existent_path = "/nonexistent/path/to/model"
        with pytest.raises(ValueError) as exc_info:
            _load_sat_model(non_existent_path)
        assert "does not exist or is not a directory" in str(exc_info.value)
        # Document creation should also fail
        with pytest.raises(ValueError):
            Document(
                raw_text="Sample text",
                paragraph_segmentation_mode="sat",
                sat_model_id=non_existent_path,
            )

        # Test file path (not a directory)
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError) as exc_info:
                _load_sat_model(temp_file.name)
            assert "does not exist or is not a directory" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(ValueError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=temp_file.name,
                )

        # Test valid path but invalid model
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(RuntimeError) as exc_info:
                _load_sat_model(temp_dir)
            assert "does not contain a valid SaT model" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(RuntimeError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=temp_dir,
                )

    @pytest.mark.skipif(
        sys.platform == "win32" and sys.version_info < (3, 12),
        reason=(
            "On Windows, Python 3.11 launches `cmd /c ver` for `platform.system()`, "
            "which aiohttp calls on import, and the network guard that the new "
            "Python process inherits refuses to launch it"
        ),
    )
    def test_local_sat_model_without_network_access(self, tmp_path):
        """
        Tests that a SaT model in a local directory that also has the tokenizer
        of the model is loaded without network access, and without the cache of
        Hugging Face.

        The model is loaded in a new Python process, which has an empty cache,
        and which the network guard of the test also applies to: in this process,
        wtpsplit-lite keeps the tokenizer that it loaded before, so the tokenizer
        would not be downloaded again here even if it were not in the directory.
        """
        model_dir = tmp_path / "sat-3l-sm"
        model_dir.mkdir()
        # The files are in the cache, where the loading of the model in the base
        # test class put them
        for repo_id, filename in (
            ("segment-any-text/sat-3l-sm", "config.json"),
            ("segment-any-text/sat-3l-sm", "model_optimized.onnx"),
            ("facebookAI/xlm-roberta-base", "tokenizer.json"),
        ):
            cached_path = try_to_load_from_cache(repo_id, filename)
            assert isinstance(cached_path, str), f"{repo_id}/{filename} not cached"
            # Where the system has symbolic links, the file in the cache is a
            # relative link to the file itself, and os.link on Linux links the
            # link, not the file (before Python 3.14): the copy would point nowhere
            cached_path = os.path.realpath(cached_path)
            try:
                # The model file is large, so it is linked rather than copied
                os.link(cached_path, model_dir / filename)
            except OSError:
                shutil.copyfile(cached_path, model_dir / filename)

        script = (
            "import sys\n"
            "from contextgem import Document\n"
            "document = Document(\n"
            "    raw_text='First sentence. Second sentence.\\n\\nNew paragraph.',\n"
            "    sat_model_id=sys.argv[1],\n"
            "    pre_segment_sentences=True,\n"
            ")\n"
            "print([len(p.sentences) for p in document.paragraphs])\n"
        )
        empty_cache_dir = tmp_path / "empty_hf_home"
        env = {
            **os.environ,
            "HF_HOME": str(empty_cache_dir),
            "HF_HUB_CACHE": str(empty_cache_dir / "hub"),
            # litellm reads its bundled model data instead of fetching it on import
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        result = subprocess.run(  # nosec B603 - the Python of the tests, no shell
            [sys.executable, "-c", script, str(model_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # Each of the two paragraphs has its sentences
        sentence_counts = ast.literal_eval(result.stdout.strip().splitlines()[-1])
        assert len(sentence_counts) == 2
        assert all(count > 0 for count in sentence_counts)
        # Nothing was taken into the cache
        assert not any(path.is_file() for path in empty_cache_dir.rglob("*"))

    def test_requires_sentence_segmentation(self):
        """
        Tests the `_requires_sentence_segmentation` method of the Document class.
        """
        # === Direct aspect/concept assignment ===

        document = Document(raw_text="This is a sentence.")

        # Aspects
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    reference_depth="paragraphs",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()
        document.remove_all_aspects()

        # Concepts
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_concepts()
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=False,
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()
        document.remove_all_concepts()
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()

        # Sub-aspects
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    aspects=[
                        Aspect(
                            name="Sub-aspect 1",
                            description="Sub-aspect 1",
                            reference_depth="sentences",
                        ),
                    ],
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    aspects=[
                        Aspect(
                            name="Sub-aspect 1",
                            description="Sub-aspect 1",
                            reference_depth="paragraphs",
                        ),
                    ],
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()

        # === Assignment of aspects/concepts via ExtractionPipeline ===

        document = Document(raw_text="This is a sentence.")

        # Aspects
        extraction_pipeline = ExtractionPipeline(
            aspects=[
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()

        # Concepts
        extraction_pipeline = ExtractionPipeline(
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert document._requires_sentence_segmentation()

    @pytest.mark.vcr
    def test_sat_model_deferred_segmentation(self):
        """
        Tests for the SaT model deferred segmentation.
        """

        # Trigger based on param
        document = Document(
            raw_text=get_test_document_text(),
            pre_segment_sentences=True,
        )
        assert document.paragraphs
        assert document.sentences

        # Trigger manually
        document = Document(raw_text=get_test_document_text())
        assert document.paragraphs
        # By default, sentences are not segmented.
        assert not document.sentences
        document._segment_sents()  # called when sentence-level refs are needed for aspects/concepts
        assert document.sentences

        # Selective segmentation of paragraphs
        document = Document(
            paragraphs=[
                Paragraph(
                    raw_text="This is sentence 1. This is sentence 2.",
                    sentences=[
                        Sentence(raw_text="This is sentence 1."),
                        Sentence(raw_text="This is sentence 2."),
                    ],
                ),
                Paragraph(
                    raw_text="This is a short sentence. And this is a bit longer sentence."
                ),  # this paragraph will be segmented
            ],
            pre_segment_sentences=True,
        )
        assert len(document.paragraphs[1].sentences) == 2
        assert len(document.sentences) == 4

        # === Segmentation is triggered when LLM extraction method is called for aspects/concepts
        # that require sentence segmentation ===

        # Some aspects require sentence segmentation (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            aspects=[
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ],
        )
        assert not document.sentences

        # Some concepts require sentence segmentation (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        assert not document.sentences

        # Some sub-aspects require sentence segmentation (assignment during Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.aspects = [
            Aspect(
                name="Liability",
                description="Liability",
                aspects=[
                    Aspect(
                        name="Liability cap",
                        description="Total liability cap",
                        reference_depth="sentences",
                    ),
                ],
            ),
        ]
        assert not document.sentences

        # Mixed - some aspects and concepts require sentence segmentation, some do not
        # (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            aspects=[
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    reference_depth="paragraphs",
                ),
                Aspect(
                    name="Aspect 2", description="Aspect 2", reference_depth="sentences"
                ),
            ],
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
                StringConcept(
                    name="Concept 2",
                    description="Concept 2",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        assert not document.sentences

        # Some aspects require sentence segmentation (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.aspects = [
            Aspect(
                name="Liability", description="Liability", reference_depth="sentences"
            ),
        ]
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_aspects_from_document(document)
        assert document.sentences

        # Some concepts require sentence segmentation (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.concepts = [
            StringConcept(
                name="Liability cap",
                description="Liability cap",
                add_references=True,
                reference_depth="sentences",
            ),
        ]
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_concepts_from_document(document)

        # Some sub-aspects require sentence segmentation (assignment during Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Liability",
                    aspects=[
                        Aspect(
                            name="Liability cap",
                            description="Total liability cap",
                            reference_depth="sentences",
                        ),
                    ],
                ),
            ]
        )
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_aspects_from_document(document)
        assert document.sentences

        # Mixed - some aspects and concepts require sentence segmentation, some do not
        # (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Confidentiality",
                    description="Confidentiality",
                    reference_depth="paragraphs",
                ),
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ]
        )
        document.add_concepts(
            [
                StringConcept(
                    name="Confidential information",
                    description="Confidential information",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_all(document)
        assert document.sentences

        # Test "from_aspects" and "from_concepts" methods
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document.sentences
        self.llm_extractor.extract_aspects_from_document(
            document, from_aspects=[document.aspects[0]]
        )
        assert document.sentences
        document = Document(raw_text=get_test_document_text())
        document.add_concepts(
            [
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                )
            ]
        )
        assert not document.sentences
        self.llm_extractor.extract_concepts_from_document(
            document, from_concepts=[document.concepts[0]]
        )
        assert document.sentences

        # === Assignment via ExtractionPipeline ===

        document = Document(raw_text=get_test_document_text())
        extraction_pipeline = ExtractionPipeline(
            aspects=[
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ],
            concepts=[
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert not document.sentences
        self.llm_extractor.extract_all(document)
        assert document.sentences

    def test_init_paragraph(self):
        """
        Tests for constructing a Paragraph instance and attaching it to a Document instance.
        """
        paragraph = Paragraph(raw_text="Test paragraph")
        assert paragraph.raw_text
        document = Document(paragraphs=[paragraph])
        assert document.raw_text  # to be populated from paragraphs
        with pytest.raises(ValueError):
            Paragraph()  # ty: ignore[missing-argument]
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="Test paragraph",
                additional_context=1,  # type: ignore
            )
        sentence = Sentence(raw_text="Test sentence")
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="XYZ",
                sentences=[sentence],  # not matched in paragraph text
            )
        with pytest.raises(ValueError):
            Paragraph(
                raw_text=sentence.raw_text,
                sentences=[sentence, sentence],  # sentences with the same ID
            )
        # Warning is logged if linebreaks occur in additional context
        Paragraph(
            raw_text="Test paragraph",
            additional_context="""
            Test
            linebreaks
            """,
        )
        # Test with non-empty text but containing only control chars
        with pytest.raises(ValueError, match="control characters"):
            Paragraph(raw_text=" \u200c ")  # zero-width non-joiner

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(paragraph)

    def test_init_sentence(self):
        """
        Tests for constructing a Sentence instance and attaching it to a Paragraph instance.
        """
        sentence = Sentence(raw_text="Test sentence")
        assert sentence.raw_text
        Paragraph(raw_text=sentence.raw_text, sentences=[sentence])
        # Warning is logged if linebreaks occur in additional context
        with pytest.raises(ValueError):
            Sentence()  # ty: ignore[missing-argument]
        Sentence(
            raw_text="Test sentence",
            additional_context="""
            Test
            linebreaks
            """,
        )
        # Test with non-empty text but containing only control chars
        with pytest.raises(ValueError, match="control characters"):
            Sentence(raw_text=" \u200c ")  # zero-width non-joiner

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(sentence)

    def test_get_paragraph_and_sentence_index(self):
        """
        Tests for looking up paragraph/sentence positions in a document by object,
        covering duplicate texts, unsupported in-place list mutations, lazy
        sentence population, empty documents, and serialization round-trips.
        """
        # Paragraph lookup, with duplicate paragraph texts (adjacent and
        # separated) resolving to their specific occurrences
        para_texts = [
            "Duplicate clause.",
            "Duplicate clause.",
            "Intro.",
            "Duplicate clause.",
            "End.",
        ]
        document = Document(paragraphs=[Paragraph(raw_text=i) for i in para_texts])
        for idx, paragraph in enumerate(document.paragraphs):
            assert document.get_paragraph_index(paragraph) == idx
            # Repeated lookups are consistent
            assert document.get_paragraph_index(paragraph) == idx
        # Sorting references by position works with duplicate texts
        refs = [
            document.paragraphs[3],
            document.paragraphs[0],
            document.paragraphs[1],
        ]
        assert sorted(refs, key=document.get_paragraph_index) == [
            document.paragraphs[0],
            document.paragraphs[1],
            document.paragraphs[3],
        ]
        # Single-paragraph document
        single_para_document = Document(paragraphs=[Paragraph(raw_text="Only one.")])
        assert (
            single_para_document.get_paragraph_index(single_para_document.paragraphs[0])
            == 0
        )
        # Foreign paragraph is not found, even with identical text
        with pytest.raises(ValueError, match="not found"):
            document.get_paragraph_index(Paragraph(raw_text="Duplicate clause."))
        # Paragraphs of a same-text twin document are not found either
        # (lookup is keyed by unique ID, not text equality)
        twin_document = Document(paragraphs=[Paragraph(raw_text=i) for i in para_texts])
        with pytest.raises(ValueError, match="not found"):
            document.get_paragraph_index(twin_document.paragraphs[0])
        # Document without paragraphs (image-only)
        image_document = Document(images=[self.test_img_png_invoice])
        with pytest.raises(ValueError, match="not found"):
            image_document.get_paragraph_index(document.paragraphs[0])
        with pytest.raises(ValueError, match="not found"):
            image_document.get_sentence_index(Sentence(raw_text="Sent A."))

        # Unsupported in-place mutations of the paragraphs list must not
        # yield stale positions (lookups always reflect the live list state)
        mutated_document = Document(
            paragraphs=[Paragraph(raw_text=i) for i in ["P0.", "P1.", "P2."]]
        )
        first_para = mutated_document.paragraphs[0]
        last_para = mutated_document.paragraphs[-1]
        assert mutated_document.get_paragraph_index(last_para) == 2
        # Insertion shifts positions
        inserted_para = Paragraph(raw_text="Inserted.")
        mutated_document.paragraphs.insert(0, inserted_para)
        assert mutated_document.get_paragraph_index(first_para) == 1
        assert mutated_document.get_paragraph_index(last_para) == 3
        assert mutated_document.get_paragraph_index(inserted_para) == 0
        # Removed paragraph no longer resolves
        del mutated_document.paragraphs[-1]
        with pytest.raises(ValueError, match="not found"):
            mutated_document.get_paragraph_index(last_para)
        assert mutated_document.get_paragraph_index(first_para) == 1
        # Removal from the start shifts positions
        del mutated_document.paragraphs[0]
        assert mutated_document.get_paragraph_index(first_para) == 0

        # Sentence lookup, with duplicate sentence texts within and
        # across paragraphs resolving to their specific occurrences
        document_with_sents = Document(
            paragraphs=[
                Paragraph(
                    raw_text="Sent A. Sent A. Sent B.",
                    sentences=[
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent B."),
                    ],
                ),
                Paragraph(
                    raw_text="Sent A. Sent C.",
                    sentences=[
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent C."),
                    ],
                ),
            ]
        )
        for para_idx, paragraph in enumerate(document_with_sents.paragraphs):
            for sent_idx, sentence in enumerate(paragraph.sentences):
                assert document_with_sents.get_sentence_index(sentence) == (
                    para_idx,
                    sent_idx,
                )
                # Repeated lookups are consistent
                assert document_with_sents.get_sentence_index(sentence) == (
                    para_idx,
                    sent_idx,
                )
        # Positions sort in document order across paragraphs, matching the
        # flattened `document.sentences` order
        all_sents = document_with_sents.sentences
        assert (
            sorted(all_sents, key=document_with_sents.get_sentence_index) == all_sents
        )
        # Foreign sentence is not found, even with identical text
        with pytest.raises(ValueError, match="not found"):
            document_with_sents.get_sentence_index(Sentence(raw_text="Sent A."))

        # Lookup error hints that sentences are not segmented yet
        with pytest.raises(ValueError, match="no sentences segmented"):
            document.get_sentence_index(Sentence(raw_text="Intro."))

        # Sentences populated after earlier lookups resolve correctly,
        # including partial (per-paragraph) population
        document.paragraphs[2].sentences = [Sentence(raw_text="Intro.")]
        assert document.get_sentence_index(document.paragraphs[2].sentences[0]) == (
            2,
            0,
        )
        document.paragraphs[4].sentences = [Sentence(raw_text="End.")]
        assert document.get_sentence_index(document.paragraphs[4].sentences[0]) == (
            4,
            0,
        )
        # Previously populated sentences still resolve
        assert document.get_sentence_index(document.paragraphs[2].sentences[0]) == (
            2,
            0,
        )
        # Once some sentences exist, the not-segmented hint no longer applies
        with pytest.raises(ValueError, match="not found") as exc_info:
            document.get_sentence_index(Sentence(raw_text="Intro."))
        assert "no sentences segmented" not in str(exc_info.value)

        # Unsupported in-place mutations of a paragraph's sentences list must not
        # yield stale positions (lookups always reflect the live list state)
        mutable_sents_para = document_with_sents.paragraphs[1]
        tracked_sent = mutable_sents_para.sentences[-1]  # "Sent C." at (1, 1)
        assert document_with_sents.get_sentence_index(tracked_sent) == (1, 1)
        # Insertion shifts positions within the paragraph
        inserted_sent = Sentence(raw_text="Sent A.")
        mutable_sents_para.sentences.insert(0, inserted_sent)
        assert document_with_sents.get_sentence_index(tracked_sent) == (1, 2)
        assert document_with_sents.get_sentence_index(inserted_sent) == (1, 0)
        # Removed sentence no longer resolves
        del mutable_sents_para.sentences[-1]  # removes tracked_sent
        with pytest.raises(ValueError, match="not found"):
            document_with_sents.get_sentence_index(tracked_sent)
        # Other paragraphs' sentences are unaffected
        assert document_with_sents.get_sentence_index(
            document_with_sents.paragraphs[0].sentences[0]
        ) == (0, 0)

        # Unique IDs survive serialization round-trips, so positions resolve
        # across original/deserialized document boundaries, in both directions
        deserialized_document = Document.from_dict(document.to_dict())
        for idx in (0, 1, 3):
            assert (
                deserialized_document.get_paragraph_index(document.paragraphs[idx])
                == idx
            )
            assert (
                document.get_paragraph_index(deserialized_document.paragraphs[idx])
                == idx
            )
        assert deserialized_document.get_sentence_index(
            document.paragraphs[4].sentences[0]
        ) == (4, 0)
        assert document.get_sentence_index(
            deserialized_document.paragraphs[2].sentences[0]
        ) == (2, 0)
        # Deserialized document's own objects resolve within itself
        for idx, paragraph in enumerate(deserialized_document.paragraphs):
            assert deserialized_document.get_paragraph_index(paragraph) == idx

        # Performing lookups (which build internal position caches) must not
        # affect document equality, serialization, or cloning
        assert Document.from_dict(document.to_dict()) == document
        assert Document.from_dict(document_with_sents.to_dict()) == document_with_sents
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.parametrize("context", [document, extraction_pipeline])
    def test_init_document_and_extraction_pipeline(
        self, context: Document | ExtractionPipeline
    ):
        """
        Tests different initialization scenarios and validations associated with
        the `Document` and `ExtractionPipeline` classes.
        """
        context = context.clone()  # clone for method-scoped state modification

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(context)

        # Document initialization
        if isinstance(context, Document):
            for sat_model_id in [
                "sat-3l-sm",
                # "sat-6l-sm",
                # "sat-12l-sm",
            ]:
                for lang in ["en", "ua", "zh"]:  # EN and other langs
                    document = Document(
                        raw_text=get_test_document_text(lang=lang),
                        sat_model_id=sat_model_id,
                        paragraph_segmentation_mode="sat",
                        pre_segment_sentences=True,
                    )
                    assert document.paragraphs  # to be segmented from text
                    assert all(
                        i.sentences for i in document.paragraphs
                    )  # segmented from paragraphs since `pre_segment_sentences` is True
            Document(
                raw_text="Random text",
                paragraphs=[
                    Paragraph(raw_text="Random text"),
                ],
            )
            document = Document(
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                    Paragraph(raw_text="Random text 3"),
                ],
            )
            assert document.raw_text  # to be populated from paragraphs
            assert not any(
                i.sentences for i in document.paragraphs
            )  # sentences not segmented yet since `pre_segment_sentences` is False (default)
            with pytest.raises(ValueError):
                document.raw_text = "Random text 1"  # cannot be set once populated
            with pytest.raises(ValueError):
                document.paragraphs = [
                    Paragraph(raw_text="Random text 1"),
                ]  # cannot be set once populated
            Document(
                images=[
                    self.test_img_png_invoice,
                ]
            )
            with pytest.raises(ValueError):
                Document()
            with pytest.raises(ValueError):
                # Paragraphs do not match the text
                Document(
                    raw_text="XYZ",
                    paragraphs=[
                        Paragraph(raw_text="Random text 1"),
                        Paragraph(raw_text="Random text 2"),
                        Paragraph(raw_text="Random text 3"),
                    ],
                )
            with pytest.raises(ValueError):
                # Paragraphs are not ordered according to their appearance in the text
                Document(
                    raw_text="AAA BBB CCC",
                    paragraphs=[
                        Paragraph(raw_text="CCC"),
                        Paragraph(raw_text="BBB"),
                        Paragraph(raw_text="AAA"),
                    ],
                )
            # Paragraphs are ordered according to their appearance in the text,
            # but some paragraphs are duplicated
            Document(
                raw_text="AAA BBB CCC AAA DDD EEE BBB FFF",
                paragraphs=[
                    Paragraph(raw_text="AAA"),
                    Paragraph(raw_text="BBB"),
                    Paragraph(raw_text="CCC"),
                    Paragraph(raw_text="AAA"),
                    Paragraph(raw_text="DDD"),
                    Paragraph(raw_text="EEE"),
                    Paragraph(raw_text="BBB"),
                    Paragraph(raw_text="FFF"),
                ],
            )
            with pytest.raises(ValueError):
                # Paragraph duplicates exist but not all duplicates are matched
                Document(
                    raw_text="AAA BBB CCC",
                    paragraphs=[
                        Paragraph(raw_text="AAA"),
                        Paragraph(raw_text="BBB"),
                        Paragraph(raw_text="CCC"),
                        Paragraph(raw_text="BBB"),
                    ],
                )
            with pytest.raises(ValueError):
                Document(
                    raw_text="Random text",
                    extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
                )
            Document(
                raw_text="Random text 1\n\nRandom text 2",
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                ],
                images=[
                    self.test_img_png_invoice,
                ],
            )
            # List field items' unique IDs
            para = Paragraph(raw_text="Random text")
            with pytest.raises(ValueError):
                Document(paragraphs=[para, para, para])
            with pytest.raises(ValueError):
                Document(
                    images=[
                        self.test_img_png_invoice,
                        self.test_img_png_invoice,
                    ]
                )
            # Test with non-empty text but containing only control chars
            with pytest.raises(ValueError, match="control characters"):
                Document(raw_text=" \u200c ")  # zero-width non-joiner

        # Extraction pipeline initialization
        elif isinstance(context, ExtractionPipeline):
            ExtractionPipeline()  # works as we can interactive add aspects and concepts after initialization
            # Pipeline assignment
            document = self.document.clone()
            document.assign_pipeline(context)
            with pytest.raises(ValueError, match="already has aspects"):
                document.assign_pipeline(
                    context
                )  # document already has aspects and concepts
            document.aspects = []
            document.concepts = []
            document.assign_pipeline(context)
            assert context.aspects is not document.aspects
            assert context.concepts is not document.concepts
            # Pipeline params
            with pytest.raises(ValueError):
                ExtractionPipeline(
                    aspects=[
                        Aspect(
                            name="Liability",
                            description="Clauses describing liability of the parties",
                        )
                    ],
                    extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
                )
            ExtractionPipeline(
                aspects=[
                    Aspect(
                        name="Liability",
                        description="Clauses describing liability of the parties",
                        add_justifications=True,
                    )
                ]
            )
            ExtractionPipeline(
                concepts=[
                    StringConcept(
                        name="Business Information",
                        description="Categories of Business Information",
                        add_justifications=True,
                    ),
                ]
            )

        # Document and extraction pipeline initialization
        # Document aspects
        document_aspects = [
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
            ),
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                add_justifications=True,
            ),
        ]
        context.aspects = document_aspects
        assert context.aspects is not document_aspects

        # Validate assignment
        with pytest.raises(ValueError, match="non-empty list"):
            context.add_aspects([])  # empty list
        with pytest.raises(ValueError, match="non-empty list"):
            context.add_concepts([])  # empty list
        with pytest.raises(ValueError):
            context.add_aspects(
                [  # ty: ignore[invalid-argument-type]
                    "Random string",
                ]  # invalid aspect type
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="liability",
                        description="clauses describing liability of the parties",
                        add_justifications=True,
                    )
                ]  # duplicate aspect (case-insensitive)
            )
        # Invalid sequence types
        with pytest.raises(ValueError, match="list"):
            context.add_concepts(
                (
                    StringConcept(
                        name="Random",
                        description="Random",
                    ),
                )  # tuple instead of list  # type: ignore
            )
        with pytest.raises(ValueError, match="list"):
            context.concepts = (
                StringConcept(
                    name="Random",
                    description="Random",
                ),
            )  # tuple instead of list
        with pytest.raises(ValueError, match="list"):
            context.concepts = {1, 2, 3}  # set instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            context.concepts = range(10)  # range instead of list  # type: ignore

        assert context.aspects == document_aspects
        assert context.aspects is not document_aspects
        context.get_aspect_by_name("Liability")
        with pytest.raises(ValueError):
            context.get_aspect_by_name("Non-existent")
        # Document concepts
        document_concepts = [
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
                add_justifications=True,
            ),
            BooleanConcept(
                name="Contract type check",
                description="Contract is an NDA",
                modality="image",
                add_justifications=True,
            ),
        ]
        context.concepts = document_concepts
        assert context.concepts is not document_concepts
        # Validate assignment
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [  # ty: ignore[invalid-assignment]
                123
            ]  # invalid concept type
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [
                StringConcept(
                    name="business information",
                    description="categories of business information",
                ),
            ]  # duplicate concept (case-insensitive)
        assert context.concepts == document_concepts
        assert context.concepts is not document_concepts
        context.get_concept_by_name("Business Information")
        with pytest.raises(ValueError):
            context.get_concept_by_name("Non-existent")

        # Adding instances
        context.add_aspects(
            [
                Aspect(
                    name="Penalties",
                    description="Clauses describing contractual penalties",
                    add_justifications=True,
                    model_role="reasoner",
                ),
                Aspect(
                    name="Term and duration",
                    description="Clauses describing term and duration of the contract",
                ),
            ]
        )
        assert context.get_aspects_by_names(["Penalties", "Term and duration"])
        context.remove_aspects_by_names(["Penalties", "Term and duration"])
        context.add_concepts(
            [
                BooleanConcept(
                    name="NDA type check",
                    description="Is NDA one-way or mutual?",
                )
            ]
        )
        assert context.get_concepts_by_names(["NDA type check"])
        context.remove_concepts_by_names(["NDA type check"])

        # Removal of instances
        # Aspects
        context.remove_aspect_by_name("Liability")
        assert len(context.aspects) == 1
        assert len(document_aspects) == 2
        with pytest.raises(ValueError):
            context.get_aspect_by_name("Liability")  # not assigned aspect
        context.remove_all_aspects()
        assert len(context.aspects) == 0
        assert len(document_aspects) == 2
        # Concepts
        context.remove_concept_by_name("Contract type check")
        assert len(context.concepts) == 1
        assert len(document_concepts) == 2
        with pytest.raises(ValueError):
            context.get_concept_by_name("Contract type check")  # not assigned concept
        context.remove_all_concepts()
        assert len(context.concepts) == 0
        assert len(document_concepts) == 2
        # All instances
        context.aspects = document_aspects
        context.concepts = document_concepts
        context.remove_all_instances()
        assert len(context.aspects) == 0
        assert len(document_aspects) == 2
        assert len(context.concepts) == 0
        assert len(document_concepts) == 2

        # List field items' unique IDs
        context.remove_all_instances()
        assert not context.aspects and not context.concepts
        aspect = Aspect(
            name="Test",
            description="Test",
        )
        with pytest.raises(ValueError):
            context.add_aspects([aspect, aspect])
        concept = StringConcept(
            name="Test",
            description="Test",
        )
        with pytest.raises(ValueError):
            context.add_concepts([concept, concept])
