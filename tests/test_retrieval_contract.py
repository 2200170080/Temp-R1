"""Exercise the actual server definitions with deterministic FAISS/encoder doubles.

No model, index, service, GPU, or package installation is required. Optional
FastAPI/TestClient checks run only when their dependencies are already available.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from typing import Dict, List, Optional

import numpy as np


SERVER = Path(__file__).resolve().parents[1] / "search_r1/search/retrieval_server.py"
DEFINITIONS = {"load_docs", "BaseRetriever", "DenseRetriever", "Config", "retrieve_endpoint"}
HAS_HTTP = all(importlib.util.find_spec(name) for name in ("fastapi", "pydantic", "httpx"))


class FakeIndex:
    def __init__(self, size):
        self.ntotal = size
        self.calls = []
        self.invalid_index = None

    def search(self, embeddings, k):
        if not 0 < k <= self.ntotal:
            raise AssertionError("FAISS received an uncapped/nonpositive topk")
        self.calls.append(k)
        rows = [np.roll(np.arange(self.ntotal), -int(row[0]))[:k] for row in embeddings]
        if self.invalid_index is not None:
            rows[0][0] = self.invalid_index
        indices = np.array(rows, dtype=np.int64)
        scores = np.tile(np.arange(k, 0, -1, dtype=np.float32), (len(rows), 1))
        return scores, indices


class FakeEncoder:
    calls = []

    def __init__(self, **kwargs):
        pass

    def encode(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            raise AssertionError("An empty query batch must not reach the encoder")
        self.calls.append(list(queries))
        return np.array([[0 if query == "first" else 1] for query in queries], dtype=np.float32)


def make_fixture(size=3, corpus_size=None, http=False):
    corpus_size = size if corpus_size is None else corpus_size
    corpus = [
        {"id": str(i), "contents": "Peter_Pace Make_a_visit China on 2007-03-22",
         "subject": "Peter_Pace", "predicate": "Make_a_visit", "object": "China", "date": "2007-03-22"}
        for i in range(corpus_size)
    ]
    index = FakeIndex(size)
    FakeEncoder.calls = []
    gpu_calls = []

    def unexpected_gpu():
        gpu_calls.append(True)
        raise AssertionError("An invalid index/corpus must fail before GPU initialization")

    namespace = {
        "__name__": "retrieval_contract_fixture", "List": List, "Dict": Dict, "Optional": Optional,
        "faiss": types.SimpleNamespace(read_index=lambda path: index, StandardGpuResources=unexpected_gpu),
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        "load_corpus": lambda path: corpus, "Encoder": FakeEncoder, "QueryRequest": object,
    }
    source = ast.parse(SERVER.read_text(encoding="utf-8"), filename=str(SERVER))
    names = DEFINITIONS | ({"QueryRequest"} if http else set())
    selected = []
    for node in source.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef) and not http:
                node.decorator_list = []
            selected.append(node)
    if http:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
        module = types.ModuleType(namespace["__name__"])
        namespace.update({"app": FastAPI(), "BaseModel": BaseModel, "Field": Field})
        module.__dict__.update(namespace)
        namespace = module.__dict__
        sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SERVER), "exec"), namespace)
    config = namespace["Config"](retrieval_method="e5", retrieval_topk=5, faiss_gpu=False,
                                  retrieval_batch_size=1)
    namespace["config"] = config
    return namespace, corpus, index, gpu_calls


class RetrievalContractTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.corpus, self.index, self.gpu_calls = make_fixture()
        self.backend = self.ns["DenseRetriever"](self.ns["config"])
        self.ns["retriever"] = self.backend

    def request(self, queries, return_scores=False, topk=None):
        request = types.SimpleNamespace(queries=queries, return_scores=return_scores, topk=topk)
        return self.ns["retrieve_endpoint"](request)["result"]

    def test_default_false_single_query_preserves_all_document_fields(self):
        result = self.request(["first"])
        self.assertEqual(result, [self.corpus])
        self.assertEqual(self.index.calls, [3])

    def test_false_multiple_queries_has_one_group_per_query(self):
        result = self.request(["first", "second"], topk=2)
        self.assertEqual(result, [self.corpus[:2], [self.corpus[1], self.corpus[2]]])

    def test_true_single_and_multiple_queries_wraps_scores(self):
        for queries in (["first"], ["first", "second"]):
            with self.subTest(queries=queries):
                result = self.request(queries, return_scores=True, topk=2)
                self.assertEqual(len(result), len(queries))
                self.assertEqual(len(result[0]), 2)
                self.assertEqual(result[0][0], {"document": self.corpus[0], "score": 2.0})

    def test_empty_queries_both_response_modes(self):
        for scores in (False, True):
            self.assertEqual(self.request([], return_scores=scores), [])
        self.assertEqual(FakeEncoder.calls, [])

    def test_search_caps_large_topk(self):
        docs, scores = self.backend.search("first", num=30, return_score=True)
        self.assertEqual(docs, self.corpus)
        self.assertEqual(scores, [3.0, 2.0, 1.0])
        self.assertEqual(self.index.calls, [3])

    def test_batch_search_caps_large_topk_and_chunks_using_capped_size(self):
        docs, scores = self.backend.batch_search(["first", "second"], num=30, return_score=True)
        self.assertEqual(docs, [self.corpus, self.corpus[1:] + self.corpus[:1]])
        self.assertEqual(scores, [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
        self.assertEqual(self.index.calls, [3, 3])

    def test_direct_empty_batch_preserves_backend_contract(self):
        self.assertEqual(self.backend.batch_search([], return_score=True), ([], []))
        self.assertEqual(self.backend.batch_search([], return_score=False), [])

    def test_nonpositive_topk_is_rejected_before_encoding(self):
        for number in (0, -1):
            with self.subTest(number=number):
                with self.assertRaisesRegex(ValueError, "positive"):
                    self.backend.search("first", num=number)
                with self.assertRaisesRegex(ValueError, "positive"):
                    self.backend.batch_search(["first"], num=number)
        self.assertEqual(FakeEncoder.calls, [])

    def test_nonpositive_config_topk_is_rejected(self):
        self.ns["config"].retrieval_topk = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            self.ns["DenseRetriever"](self.ns["config"])

    def test_load_docs_rejects_negative_and_out_of_range(self):
        for index in (-1, len(self.corpus)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                self.ns["load_docs"](self.corpus, [index])
        self.assertEqual(self.ns["load_docs"](self.corpus, ["1"]), [self.corpus[1]])

    def test_unexpected_faiss_sentinel_fails_instead_of_selecting_last_document(self):
        self.index.invalid_index = -1
        with self.assertRaises(IndexError):
            self.backend.search("first", num=2)
        with self.assertRaises(IndexError):
            self.backend.batch_search(["first"], num=2)

    def test_empty_index_fails_before_gpu_or_encoder_initialization(self):
        ns, _, _, gpu_calls = make_fixture(size=0)
        ns["config"].faiss_gpu = True
        with self.assertRaisesRegex(ValueError, "index is empty"):
            ns["DenseRetriever"](ns["config"])
        self.assertEqual(gpu_calls, [])
        self.assertEqual(FakeEncoder.calls, [])

    def test_row_count_mismatch_fails_before_gpu_or_encoder_initialization(self):
        ns, _, _, gpu_calls = make_fixture(size=3, corpus_size=2)
        ns["config"].faiss_gpu = True
        with self.assertRaisesRegex(ValueError, "index=3, corpus=2"):
            ns["DenseRetriever"](ns["config"])
        self.assertEqual(gpu_calls, [])
        self.assertEqual(FakeEncoder.calls, [])

@unittest.skipUnless(HAS_HTTP, "FastAPI, Pydantic, and httpx are not installed; no HTTP test was run")
class OptionalHttpContractTests(unittest.TestCase):
    def test_http_request_and_response_contract(self):
        from fastapi.testclient import TestClient
        ns, corpus, _, _ = make_fixture(http=True)
        ns["retriever"] = ns["DenseRetriever"](ns["config"])
        with TestClient(ns["app"]) as client:
            for queries in ([], ["first"], ["first", "second"]):
                for scores in (False, True):
                    response = client.post("/retrieve", json={"queries": queries, "return_scores": scores, "topk": 30})
                    self.assertEqual(response.status_code, 200)
                    groups = response.json()["result"]
                    self.assertEqual(len(groups), len(queries))
                    if queries:
                        self.assertEqual(len(groups[0]), len(corpus))
            self.assertEqual(client.post("/retrieve", json={"queries": ["first"]}).json(), {"result": [corpus]})
            for topk in (0, -1, True, False, 1.5, "3"):
                self.assertEqual(client.post("/retrieve", json={"queries": ["first"], "topk": topk}).status_code, 422)
            for payload in ({}, {"queries": "first"}, {"queries": [None]}):
                self.assertEqual(client.post("/retrieve", json=payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
