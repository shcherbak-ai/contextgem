# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Prices the requests recorded in the VCR cassettes of the tests, from the usage
that each recorded response carries, i.e. without sending a request.

Each request is priced on its own, as the library prices it: with genai-prices,
under the provider of the decision requests where genai-prices has one, and
with the price set here for a model that genai-prices has no price for. The
prices are the current ones, of the version of genai-prices that the project
pins, so the cost of a cassette is what recording it again would cost, which can
differ from what it cost when it was recorded. The requests to local models are
free. A response without usage, e.g. an error or the answer to a request for
the metadata of a local model, is not counted.

Run it from the root of the repo:

```bash
# All the cassettes
uv run python -m tests.cassette_costs

# The cassettes that differ from the last commit, i.e. those recorded since
uv run python -m tests.cassette_costs --changed

# Some cassettes
uv run python -m tests.cassette_costs tests/cassettes/<name>.yaml
```

A test run reports at its end what the requests of the cassettes it recorded
cost, and what those of the cassettes it replayed would cost if they were sent
live (see ``pytest_terminal_summary`` in ``tests/conftest.py``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess  # nosec B404 - runs git with arguments of its own, no shell
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from genai_prices import Usage as _GPUsage
from genai_prices import calc_price as _calc_auto_price
from yaml.cyaml import CSafeLoader

from contextgem.internal.base.decision_models import _DECISION_PRICING_PROVIDER_IDS
from tests.conftest import VCR_DUMMY_ENDPOINT_PREFIX


_REPO_DIR = Path(__file__).parent.parent
CASSETTES_DIR = Path(__file__).parent / "cassettes"

# The provider of the API at each host that the cassettes have requests to, as
# in the identifiers of the models
_HOST_PROVIDERS = {
    "api.openai.com": "openai",
    "api.typesafe.ai": "typesafe",
    # The host that the host of a Microsoft Foundry resource is replaced with
    urlsplit(VCR_DUMMY_ENDPOINT_PREFIX).hostname: "azure_ai",
}

# The hosts of the local servers of models, whose requests are free, and what
# the report names their provider
_LOCAL_HOSTS = ("localhost", "127.0.0.1")
_LOCAL = "local"

# The ends of the paths of the decision requests: the Decisions API of OpenAI,
# and the route of litellm for the other providers of decision models. The other
# requests to the API of a model are chat requests.
_DECISION_PATH_ENDS = ("/decisions", "/systemone")

# Prices of 1M input tokens, in USD, of the models that genai-prices has no price
# for, by their providers and models as the responses name them. Decision models
# only, whose output tokens are free: the price is the one that the evaluation
# harness sets for the provider (`_DECISION_INPUT_PRICES` in dev/evals/harness.py).
_SET_INPUT_PRICES = {
    ("azure_ai", "microsoft-decision-1"): Decimal("0.042"),
}

# How the requests of a model are priced, as the report names it
_PRICED_BY_GENAI_PRICES = "genai-prices"
_PRICED_BY_SET_PRICE = "set here"
_PRICED_AS_LOCAL = "local, free"
_NOT_PRICED = "no price"


@dataclass
class ModelCost:
    """
    What the recorded requests of one model, at one kind of endpoint, cost.

    :ivar priced_by: How the requests are priced: by genai-prices, by the price
        set here, as requests to a local model, or not at all.
    :vartype priced_by: str
    :ivar requests: The number of requests.
    :vartype requests: int
    :ivar input_tokens: The input tokens, including the cached ones.
    :vartype input_tokens: int
    :ivar cache_read_tokens: The input tokens read from the prompt cache.
    :vartype cache_read_tokens: int
    :ivar output_tokens: The output tokens.
    :vartype output_tokens: int
    :ivar cost: The cost in USD, 0 for requests that are not priced.
    :vartype cost: Decimal
    """

    priced_by: str
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = field(default_factory=Decimal)


@dataclass
class CassetteCosts:
    """
    What the requests recorded in some cassettes cost.

    :ivar n_cassettes: The number of cassettes.
    :vartype n_cassettes: int
    :ivar by_model: The cost by model and kind of endpoint ("chat" or "decision").
        A model is named by its provider and its name in the responses, e.g.
        ``openai/gpt-6-luna``, and a local model by ``local/<name>``.
    :vartype by_model: dict[tuple[str, str], ModelCost]
    :ivar by_cassette: The cost in USD of each cassette, by its file name.
    :vartype by_cassette: dict[str, Decimal]
    :ivar n_without_usage: The number of successful responses of the API of
        a hosted model that carry no usage that can be read, which are not priced.
    :vartype n_without_usage: int
    """

    n_cassettes: int = 0
    by_model: dict[tuple[str, str], ModelCost] = field(default_factory=dict)
    by_cassette: dict[str, Decimal] = field(default_factory=dict)
    n_without_usage: int = 0

    @property
    def total(self) -> Decimal:
        """
        The cost of all the requests, in USD.

        :return: The cost.
        :rtype: Decimal
        """
        return sum((cost.cost for cost in self.by_model.values()), Decimal(0))

    @property
    def not_priced(self) -> list[str]:
        """
        The models whose requests are not priced, as no price is known for them,
        which the total leaves out.

        :return: The models, each with its kind of endpoint, e.g.
            "openai/new-model (chat)".
        :rtype: list[str]
        """
        return [
            f"{name} ({kind})"
            for (name, kind), cost in sorted(self.by_model.items())
            if cost.priced_by == _NOT_PRICED
        ]


def _get_header(headers: dict[str, list[str]] | None, name: str) -> list[str]:
    """
    Returns the values of a header of a recorded request or response.

    :param headers: The headers, by name, as the cassette has them.
    :type headers: dict[str, list[str]] | None
    :param name: The name of the header, in any case.
    :type name: str
    :return: The values of the header, in lower case, or an empty list.
    :rtype: list[str]
    """
    for key, values in (headers or {}).items():
        if key.lower() == name:
            return [str(value).lower() for value in values]
    return []


def _read_json_body(message: dict[str, Any]) -> Any:
    """
    Returns the body of a recorded request or response, read as JSON.

    A response body is recorded as it was received, e.g. compressed with gzip.

    :param message: The recorded request or response.
    :type message: dict[str, Any]
    :return: The body, or None if it is empty or is not JSON.
    :rtype: Any
    """
    body = message.get("body")
    raw = body.get("string") if isinstance(body, dict) else body
    if not raw:
        return None
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    try:
        encodings = _get_header(message.get("headers"), "content-encoding")
        if "gzip" in encodings:
            raw = gzip.decompress(raw)
        elif "deflate" in encodings:
            raw = zlib.decompress(raw)
        return json.loads(raw)
    except (OSError, EOFError, zlib.error, ValueError):
        return None


def _price_request(
    provider: str, model: str, kind: str, usage: _GPUsage
) -> tuple[Decimal, str]:
    """
    Prices a request to a hosted model.

    :param provider: The provider of the model, as in the identifiers of models.
    :type provider: str
    :param model: The model, as the response names it.
    :type model: str
    :param kind: The kind of the endpoint: "chat" or "decision".
    :type kind: str
    :param usage: The tokens of the request.
    :type usage: genai_prices.Usage
    :return: The cost in USD, and how it is priced (0 if it is not priced).
    :rtype: tuple[Decimal, str]
    """
    provider_id = (
        _DECISION_PRICING_PROVIDER_IDS.get(provider, provider)
        if kind == "decision"
        else provider
    )
    try:
        price = _calc_auto_price(usage, model_ref=model, provider_id=provider_id)
        return Decimal(str(price.total_price)), _PRICED_BY_GENAI_PRICES
    except LookupError:
        pass
    input_price = _SET_INPUT_PRICES.get((provider, model))
    if input_price is not None:
        cost = Decimal(usage.input_tokens or 0) * input_price / Decimal(1_000_000)
        return cost, _PRICED_BY_SET_PRICE
    return Decimal(0), _NOT_PRICED


def _add_interaction(
    costs: CassetteCosts, cassette_name: str, interaction: dict[str, Any]
) -> None:
    """
    Adds a recorded request to the costs, if its response carries usage.

    :param costs: The costs, which are updated.
    :type costs: CassetteCosts
    :param cassette_name: The file name of the cassette that has the request.
    :type cassette_name: str
    :param interaction: The request and the response, as the cassette has them.
    :type interaction: dict[str, Any]
    :return: None
    :rtype: None
    """
    request, response = interaction["request"], interaction["response"]
    parts = urlsplit(request["uri"])
    host = (parts.hostname or "").lower()
    provider = _LOCAL if host in _LOCAL_HOSTS else _HOST_PROVIDERS.get(host)
    if provider is None:
        # Not the API of a model, e.g. a download of the prices of genai-prices
        return
    body = _read_json_body(response)
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        if provider != _LOCAL and 200 <= response["status"]["code"] < 300:
            costs.n_without_usage += 1
        return
    request_body = _read_json_body(request)
    model = str(
        body.get("model")
        or (request_body.get("model") if isinstance(request_body, dict) else None)
        or "unknown"
    )
    kind = "decision" if parts.path.endswith(_DECISION_PATH_ENDS) else "chat"
    # The input tokens include the cached ones, in genai-prices as well
    details = (
        usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    )
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens")) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens")) or 0)
    cache_read_tokens = int(details.get("cached_tokens") or 0)
    if provider == _LOCAL:
        cost, priced_by = Decimal(0), _PRICED_AS_LOCAL
    else:
        tokens = _GPUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        cost, priced_by = _price_request(provider, model, kind, tokens)
    name = f"{provider}/{model}"
    model_cost = costs.by_model.setdefault((name, kind), ModelCost(priced_by))
    model_cost.requests += 1
    model_cost.input_tokens += input_tokens
    model_cost.cache_read_tokens += cache_read_tokens
    model_cost.output_tokens += output_tokens
    model_cost.cost += cost
    costs.by_cassette[cassette_name] += cost


def price_cassettes(paths: Iterable[str | Path]) -> CassetteCosts:
    """
    Prices the requests recorded in cassettes.

    :param paths: The paths of the cassette files.
    :type paths: Iterable[str | Path]
    :return: What the requests cost.
    :rtype: CassetteCosts
    """
    costs = CassetteCosts()
    for path in map(Path, paths):
        with path.open(encoding="utf-8") as file:
            cassette = yaml.load(file, Loader=CSafeLoader) or {}
        costs.n_cassettes += 1
        costs.by_cassette[path.name] = Decimal(0)
        for interaction in cassette.get("interactions", []):
            _add_interaction(costs, path.name, interaction)
    return costs


def format_costs(costs: CassetteCosts, title: str, top: int = 10) -> list[str]:
    """
    Formats the costs as a report: a table of the models, the total, and the most
    expensive cassettes.

    :param costs: The costs.
    :type costs: CassetteCosts
    :param title: The first line of the report, which says what was priced.
    :type title: str
    :param top: The number of the most expensive cassettes that are listed.
        Defaults to 10.
    :type top: int
    :return: The lines of the report.
    :rtype: list[str]
    """
    version = metadata.version("genai-prices")
    lines = [f"{title}, at the current prices (genai-prices {version}):", ""]
    if not costs.by_model:
        return [*lines, "No request with usage is recorded in them."]
    width = max(len("Model"), *(len(name) for name, _ in costs.by_model))
    lines.append(
        f"{'Model':<{width}}  {'Endpoint':<8}  {'Requests':>8}  {'Input':>11}  "
        f"{'Cached':>11}  {'Output':>9}  {'USD':>8}  Priced by"
    )
    rows = sorted(costs.by_model.items(), key=lambda item: (-item[1].cost, item[0]))
    for (name, kind), cost in rows:
        lines.append(
            f"{name:<{width}}  {kind:<8}  {cost.requests:>8,}  "
            f"{cost.input_tokens:>11,}  {cost.cache_read_tokens:>11,}  "
            f"{cost.output_tokens:>9,}  {cost.cost:>8.4f}  {cost.priced_by}"
        )
    n_requests = sum(cost.requests for cost in costs.by_model.values())
    lines.append(
        f"{'Total':<{width}}  {'':<8}  {n_requests:>8,}  {'':>11}  {'':>11}  "
        f"{'':>9}  {costs.total:>8.4f}"
    )
    if costs.not_priced:
        lines += ["", _format_not_priced(costs)]
    if any(cost.priced_by == _PRICED_BY_SET_PRICE for cost in costs.by_model.values()):
        lines += [
            "",
            (
                f"'{_PRICED_BY_SET_PRICE}': genai-prices has no price for the model, "
                "the price is set in tests/cassette_costs.py."
            ),
        ]
    if costs.n_without_usage:
        lines += [
            "",
            (
                f"{costs.n_without_usage} successful responses of the APIs of hosted "
                "models carry no usage that can be read, and are left out."
            ),
        ]
    expensive = sorted(costs.by_cassette.items(), key=lambda item: (-item[1], item[0]))
    expensive = [(name, cost) for name, cost in expensive[:top] if cost > 0]
    if expensive and costs.n_cassettes > 1:
        lines += ["", "Most expensive cassettes (USD):"]
        lines += [f"  {cost:.4f}  {name}" for name, cost in expensive]
    return lines


def _format_not_priced(costs: CassetteCosts) -> str:
    """
    Formats the note on the models whose requests are not priced.

    :param costs: The costs, with models that are not priced.
    :type costs: CassetteCosts
    :return: The note.
    :rtype: str
    """
    return (
        f"Not priced, as no price is known for them: {', '.join(costs.not_priced)}. "
        "The total leaves them out: set their prices in tests/cassette_costs.py."
    )


def format_replayed_costs(costs: CassetteCosts) -> str:
    """
    Formats what the requests replayed from cassettes would cost if they were sent
    live, in a line.

    :param costs: The costs of the cassettes replayed.
    :type costs: CassetteCosts
    :return: The line, which names the models without a price, if any.
    :rtype: str
    """
    n_requests = sum(cost.requests for cost in costs.by_model.values())
    line = (
        f"Replayed {count_cassettes(costs.n_cassettes)}: {n_requests:,} requests, "
        f"USD {costs.total:.4f} if they were sent live, at the current prices "
        f"(genai-prices {metadata.version('genai-prices')})."
    )
    if costs.not_priced:
        line += f" {_format_not_priced(costs)}"
    return line


def count_cassettes(n: int) -> str:
    """
    Returns a number of cassettes in words, e.g. "1 cassette" or "3 cassettes".

    :param n: The number of cassettes.
    :type n: int
    :return: The number with the noun.
    :rtype: str
    """
    return f"{n} cassette{'' if n == 1 else 's'}"


def _get_changed_cassettes() -> list[Path]:
    """
    Returns the cassettes that differ from the last commit: modified, added or
    renamed, staged or not, and untracked ones.

    :return: The paths of the cassettes.
    :rtype: list[Path]
    :raises SystemExit: If git is not found.
    """
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git was not found.")
    commands = (
        [git, "diff", "--name-only", "-z", "--diff-filter=d", "HEAD", "--"],
        [git, "ls-files", "--others", "--exclude-standard", "-z", "--"],
    )
    names: set[str] = set()
    for command in commands:
        result = subprocess.run(  # nosec B603 - git with arguments of its own, no shell
            [*command, "tests/cassettes"],
            cwd=_REPO_DIR,
            capture_output=True,
            check=True,
        )
        names.update(result.stdout.decode("utf-8").split("\0"))
    return sorted(_REPO_DIR / name for name in names if name.endswith(".yaml"))


def main() -> None:
    """
    Prices the requests recorded in the cassettes, and prints the report.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Prices the requests recorded in the VCR cassettes of the tests, "
        "from the usage in their responses. Sends no request."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="cassette files (default: every cassette of tests/cassettes)",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="price the cassettes that differ from the last commit",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of the most expensive cassettes listed (default: 10)",
    )
    args = parser.parse_args()
    if args.changed and args.paths:
        parser.error("give cassette files or --changed, not both")

    if args.changed:
        paths = _get_changed_cassettes()
        if not paths:
            print("No cassette differs from the last commit.")
            return
        title = (
            f"Requests recorded in the {count_cassettes(len(paths))} changed "
            f"since the last commit"
        )
    elif args.paths:
        paths = args.paths
        title = f"Requests recorded in {count_cassettes(len(paths))}"
    else:
        paths = sorted(CASSETTES_DIR.glob("*.yaml"))
        title = (
            f"Requests recorded in the {count_cassettes(len(paths))} of tests/cassettes"
        )
    print("\n".join(format_costs(price_cassettes(paths), title, top=args.top)))


if __name__ == "__main__":
    main()
