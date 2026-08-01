"""Bounded local llama.cpp HTTP client and inference-only rollout backend."""

from __future__ import annotations

import ipaddress
import json
import math
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from urllib import error, parse, request

from ..capability import (
    BackendCapabilities,
    Capability,
    CapabilityEvidence,
    CapabilityState,
)
from ..trajectory import RolloutRequest, Trajectory

_BACKEND_NAME = "llama_cpp_vulkan"
_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_MAX_RESPONSE_BYTES = 1_048_576
_MAX_RESPONSE_BYTES = 16_777_216
_MAX_TIMEOUT_SECONDS = 60.0
_SUPPORTED_SAMPLING_PARAMETERS = frozenset({"do_sample", "max_new_tokens", "temperature", "top_p"})


class LlamaCppServerError(RuntimeError):
    """A translated transport, response, or llama.cpp protocol failure."""


class LlamaCppServerClient:
    """A bounded standard-library client for one local llama-server endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.endpoint = _normalize_endpoint(endpoint)
        self.timeout_seconds = _bounded_timeout(timeout_seconds)
        self.max_response_bytes = _bounded_response_size(max_response_bytes)
        self._closed = False

    def health(self) -> Mapping[str, object]:
        """Return a validated llama-server health response."""
        payload = self._request_json("GET", "/health")
        status = payload.get("status")
        if status != "ok":
            raise LlamaCppServerError("llama.cpp health response status must be 'ok'")
        version = payload.get("version")
        if version is not None and (not isinstance(version, str) or not version):
            raise LlamaCppServerError("llama.cpp health response version must be a string")
        return MappingProxyType(dict(payload))

    def models(self) -> tuple[Mapping[str, object], ...]:
        """Return the non-empty validated model records reported by llama-server."""
        payload = self._request_json("GET", "/v1/models")
        raw_models = payload.get("data")
        if not isinstance(raw_models, list) or not raw_models:
            raise LlamaCppServerError("llama.cpp models response data must be a non-empty array")
        models: list[Mapping[str, object]] = []
        for index, raw_model in enumerate(raw_models):
            if not isinstance(raw_model, Mapping):
                raise LlamaCppServerError(
                    f"llama.cpp models response data[{index}] must be an object"
                )
            model = dict(raw_model)
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise LlamaCppServerError(
                    f"llama.cpp models response data[{index}].id must be a string"
                )
            metadata = model.get("meta")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise LlamaCppServerError(
                    f"llama.cpp models response data[{index}].meta must be an object"
                )
            models.append(MappingProxyType(model))
        return tuple(models)

    def completion(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """Return one validated JSON-object response from the native completion route."""
        if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
            raise TypeError("completion payload must be a string-keyed mapping")
        _validate_json_value(payload, "completion payload")
        return self._request_json("POST", "/completion", payload)

    def close(self) -> None:
        """Prevent later requests; no persistent network resource is retained."""
        self._closed = True

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        if self._closed:
            raise RuntimeError("llama.cpp server client is closed")
        data = None
        headers = {"Accept": "application/json", "Connection": "close"}
        if payload is not None:
            try:
                data = json.dumps(
                    payload,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("completion payload must contain finite JSON values") from exc
            headers["Content-Type"] = "application/json"
        server_request = request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(server_request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    raise LlamaCppServerError(
                        f"llama.cpp {path} response Content-Type must be application/json"
                    )
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError as exc:
                        raise LlamaCppServerError(
                            f"llama.cpp {path} response Content-Length is invalid"
                        ) from exc
                    if declared_length < 0 or declared_length > self.max_response_bytes:
                        raise LlamaCppServerError(
                            f"llama.cpp {path} response exceeds {self.max_response_bytes} bytes"
                        )
                raw = response.read(self.max_response_bytes + 1)
        except error.HTTPError as exc:
            raise LlamaCppServerError(f"llama.cpp HTTP {exc.code} response from {path}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LlamaCppServerError(f"llama.cpp request timed out while calling {path}") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LlamaCppServerError(
                    f"llama.cpp request timed out while calling {path}"
                ) from exc
            raise LlamaCppServerError(f"llama.cpp {path} request failed: {exc.reason}") from exc
        except OSError as exc:
            raise LlamaCppServerError(f"llama.cpp {path} request failed: {exc}") from exc
        if len(raw) > self.max_response_bytes:
            raise LlamaCppServerError(
                f"llama.cpp {path} response exceeds {self.max_response_bytes} bytes"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LlamaCppServerError(f"llama.cpp {path} response is not valid UTF-8") from exc
        try:
            decoded = json.loads(text, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            detail = "non-finite JSON" if "non-finite" in str(exc) else "valid JSON"
            raise LlamaCppServerError(f"llama.cpp {path} response is not {detail}") from exc
        _validate_json_value(decoded, f"llama.cpp {path} response")
        if not isinstance(decoded, Mapping):
            raise LlamaCppServerError(f"llama.cpp {path} response must be a JSON object")
        return MappingProxyType(dict(decoded))


@dataclass(frozen=True)
class _PreparedRollout:
    request: RolloutRequest
    request_index: int
    parameters: Mapping[str, object]
    max_new_tokens: int


class LlamaCppVulkanBackend:
    """Generate backend-neutral trajectories without claiming trainable capabilities."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.max_new_tokens = _positive_integer(max_new_tokens, "max_new_tokens")
        self.client = LlamaCppServerClient(
            endpoint,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        self._backend_version = "unknown"
        self._model_identifier = "unknown"
        self._generation_observed = False
        self._gguf_observed = False
        self._token_log_probs_observed = False
        self._closed = False

    def generate(self, requests: Sequence[RolloutRequest]) -> Sequence[Trajectory]:
        """Generate one ordered trajectory for every requested sample."""
        if self._closed:
            raise RuntimeError("llama.cpp Vulkan backend is closed")
        prepared = self._prepare_requests(requests)
        if not prepared:
            return ()
        health = self.client.health()
        models = self.client.models()
        self._record_metadata(health, models)
        trajectories: list[Trajectory] = []
        for item in prepared:
            for sample_index in range(item.request.num_samples):
                effective_seed = _effective_seed(item.request.seed, sample_index)
                payload = self._completion_payload(
                    item.request.prompt,
                    item.parameters,
                    item.max_new_tokens,
                    effective_seed,
                )
                response = self.client.completion(payload)
                content, token_ids, log_probs = self._completion_evidence(response)
                self._generation_observed = True
                if log_probs is not None:
                    self._token_log_probs_observed = True
                trajectories.append(
                    Trajectory(
                        trajectory_id=self._trajectory_id(
                            item.request,
                            item.request_index,
                            sample_index,
                        ),
                        case_id=item.request.case_id,
                        prompt=item.request.prompt,
                        response=content,
                        response_token_ids=token_ids,
                        old_log_probs=log_probs,
                        sampling_parameters={
                            "max_new_tokens": item.max_new_tokens,
                            **item.parameters,
                        },
                        backend_name=_BACKEND_NAME,
                        backend_version=self._backend_version,
                        model_identifier=self._model_identifier,
                        policy_version=item.request.policy_version,
                        seed=effective_seed,
                    )
                )
        return tuple(trajectories)

    def capabilities(self) -> BackendCapabilities:
        """Report only server behavior already observed by this backend instance."""
        unsupported = {
            Capability.SUPPORTS_BACKWARD,
            Capability.SUPPORTS_CUDA,
            Capability.SUPPORTS_DISTRIBUTED_TRAINING,
            Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
            Capability.SUPPORTS_LORA_TRAINING,
            Capability.SUPPORTS_MIXED_PRECISION,
            Capability.SUPPORTS_OPTIMIZER_STEP,
            Capability.SUPPORTS_REFERENCE_LOG_PROBS,
            Capability.SUPPORTS_VALUE_HEAD,
        }
        capabilities: dict[Capability, CapabilityEvidence] = {}
        for capability in Capability:
            if capability in unsupported:
                capabilities[capability] = CapabilityEvidence(
                    state=CapabilityState.UNSUPPORTED,
                    evidence=(
                        "The llama.cpp server boundary is inference-only and exposes no "
                        f"{capability.value} operation."
                    ),
                )
            else:
                capabilities[capability] = CapabilityEvidence(
                    state=CapabilityState.UNKNOWN,
                    evidence=f"No positive {capability.value} evidence has been observed.",
                )
        self._set_observed_capability(
            capabilities,
            Capability.SUPPORTS_GENERATION,
            self._generation_observed,
            "A validated native /completion response was observed.",
        )
        self._set_observed_capability(
            capabilities,
            Capability.SUPPORTS_GGUF,
            self._gguf_observed,
            "The llama-server model record identifies GGUF format.",
        )
        self._set_observed_capability(
            capabilities,
            Capability.SUPPORTS_TOKEN_LOG_PROBS,
            self._token_log_probs_observed,
            "A validated completion probability record was observed.",
        )
        return BackendCapabilities(
            backend_name=_BACKEND_NAME,
            backend_version=self._backend_version,
            capabilities=capabilities,
        )

    def close(self) -> None:
        """Close the client exactly once while allowing repeated close calls."""
        if not self._closed:
            self.client.close()
            self._closed = True

    def _prepare_requests(
        self,
        requests: Sequence[RolloutRequest],
    ) -> tuple[_PreparedRollout, ...]:
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a sequence of RolloutRequest values")
        prepared: list[_PreparedRollout] = []
        for request_index, rollout in enumerate(requests):
            if not isinstance(rollout, RolloutRequest):
                raise TypeError("requests must contain RolloutRequest values")
            if not isinstance(rollout.prompt, str) or not rollout.prompt.strip():
                raise ValueError("rollout prompt must not be empty")
            _positive_integer(rollout.num_samples, "num_samples")
            _optional_string(rollout.case_id, "case_id")
            _optional_string(rollout.policy_version, "policy_version")
            if rollout.seed is not None:
                _non_negative_integer(rollout.seed, "seed")
            if not isinstance(rollout.sampling_parameters, Mapping) or not all(
                isinstance(key, str) for key in rollout.sampling_parameters
            ):
                raise TypeError("sampling_parameters must be a string-keyed mapping")
            parameters = dict(rollout.sampling_parameters)
            unknown = sorted(set(parameters) - _SUPPORTED_SAMPLING_PARAMETERS)
            if unknown:
                raise ValueError(f"unsupported sampling parameters: {', '.join(unknown)}")
            max_new_tokens = _positive_integer(
                parameters.pop("max_new_tokens", self.max_new_tokens),
                "max_new_tokens",
            )
            _validate_sampling_parameters(parameters)
            prepared.append(
                _PreparedRollout(
                    request=rollout,
                    request_index=request_index,
                    parameters=MappingProxyType(parameters),
                    max_new_tokens=max_new_tokens,
                )
            )
        return tuple(prepared)

    def _record_metadata(
        self,
        health: Mapping[str, object],
        models: tuple[Mapping[str, object], ...],
    ) -> None:
        version = health.get("version")
        self._backend_version = version if isinstance(version, str) else "unknown"
        selected = models[0]
        self._model_identifier = str(selected["id"])
        metadata = selected.get("meta")
        format_name = metadata.get("format") if isinstance(metadata, Mapping) else None
        self._gguf_observed = isinstance(format_name, str) and format_name.casefold() == "gguf"

    @staticmethod
    def _completion_payload(
        prompt: str,
        parameters: Mapping[str, object],
        max_new_tokens: int,
        seed: int | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "n_predict": max_new_tokens,
            "n_probs": 1,
            "prompt": prompt,
            "stream": False,
        }
        do_sample = parameters.get("do_sample")
        if do_sample is False:
            payload["temperature"] = 0.0
        elif "temperature" in parameters:
            payload["temperature"] = parameters["temperature"]
        if "top_p" in parameters:
            payload["top_p"] = parameters["top_p"]
        if seed is not None:
            payload["seed"] = seed
        return payload

    def _completion_evidence(
        self,
        response: Mapping[str, object],
    ) -> tuple[str, tuple[int, ...] | None, tuple[float, ...] | None]:
        content = response.get("content")
        if not isinstance(content, str):
            raise LlamaCppServerError("llama.cpp completion content must be a string")
        response_model = response.get("model")
        if response_model is not None and response_model != self._model_identifier:
            raise LlamaCppServerError(
                "llama.cpp completion model does not match the selected model"
            )
        tokens = _response_tokens(response.get("tokens"))
        log_probs = _response_log_probs(response.get("completion_probabilities"))
        if tokens is not None and log_probs is not None and len(tokens) != len(log_probs):
            raise LlamaCppServerError("llama.cpp completion tokens and probabilities must align")
        return content, tokens, log_probs

    @staticmethod
    def _trajectory_id(
        request_value: RolloutRequest,
        request_index: int,
        sample_index: int,
    ) -> str:
        request_part = request_value.case_id or f"request-{request_index}"
        version_part = request_value.policy_version or "unversioned"
        return f"{request_part}-{version_part}-{sample_index}"

    @staticmethod
    def _set_observed_capability(
        capabilities: dict[Capability, CapabilityEvidence],
        capability: Capability,
        observed: bool,
        evidence: str,
    ) -> None:
        if observed:
            capabilities[capability] = CapabilityEvidence(
                state=CapabilityState.SUPPORTED,
                evidence=evidence,
            )


def _normalize_endpoint(endpoint: object) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("llama.cpp endpoint must be a non-empty string")
    try:
        parsed = parse.urlsplit(endpoint.strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError("llama.cpp endpoint is malformed") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("llama.cpp endpoint scheme must be HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("llama.cpp endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("llama.cpp endpoint must not contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("llama.cpp endpoint must not contain a path")
    hostname = parsed.hostname
    if hostname is None or not _is_loopback_host(hostname):
        raise ValueError("llama.cpp endpoint host must be local loopback")
    host = f"[{hostname}]" if ":" in hostname else hostname.casefold()
    authority = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{authority}"


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _bounded_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or not 0.0 < timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be finite and in (0, {_MAX_TIMEOUT_SECONDS:g}]")
    return timeout


def _bounded_response_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_response_bytes must be an integer")
    if not 0 < value <= _MAX_RESPONSE_BYTES:
        raise ValueError(f"max_response_bytes must be in [1, {_MAX_RESPONSE_BYTES}]")
    return value


def _validate_json_value(value: object, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for item in value:
            _validate_json_value(item, field_name)
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{field_name} contains a non-string object key")
        for item in value.values():
            _validate_json_value(item, field_name)
        return
    raise ValueError(f"{field_name} contains a non-JSON value")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _optional_string(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")


def _validate_sampling_parameters(parameters: Mapping[str, object]) -> None:
    do_sample = parameters.get("do_sample")
    if do_sample is not None and not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a boolean")
    temperature = parameters.get("temperature")
    if temperature is not None:
        _bounded_number(temperature, "temperature", lower=0.0, upper=None, lower_inclusive=True)
    top_p = parameters.get("top_p")
    if top_p is not None:
        _bounded_number(top_p, "top_p", lower=0.0, upper=1.0, lower_inclusive=False)


def _bounded_number(
    value: object,
    field_name: str,
    *,
    lower: float,
    upper: float | None,
    lower_inclusive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    lower_valid = number >= lower if lower_inclusive else number > lower
    if not math.isfinite(number) or not lower_valid or (upper is not None and number > upper):
        raise ValueError(f"{field_name} is outside the supported finite range")
    return number


def _effective_seed(seed: int | None, sample_index: int) -> int | None:
    return None if seed is None else seed + sample_index


def _response_tokens(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in value
    ):
        raise LlamaCppServerError(
            "llama.cpp completion tokens must be non-negative integer token IDs"
        )
    return tuple(value)


def _response_log_probs(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise LlamaCppServerError("llama.cpp completion probabilities must be an array")
    if not value:
        raise LlamaCppServerError("llama.cpp completion probabilities must be non-empty")
    log_probs: list[float] = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise LlamaCppServerError(
                f"llama.cpp completion probabilities[{index}] must be an object"
            )
        content = record.get("content")
        candidates = record.get("probs")
        if not isinstance(content, str) or not isinstance(candidates, list):
            raise LlamaCppServerError(
                f"llama.cpp completion probabilities[{index}] has invalid content or probs"
            )
        matches = [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping) and candidate.get("tok_str") == content
        ]
        if len(matches) != 1:
            raise LlamaCppServerError(
                f"llama.cpp completion probabilities[{index}] must identify one selected token"
            )
        probability = matches[0].get("prob")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 < probability <= 1.0
        ):
            raise LlamaCppServerError(
                f"llama.cpp completion probabilities[{index}] probability must be finite in (0, 1]"
            )
        log_probs.append(math.log(float(probability)))
    return tuple(log_probs)


__all__ = [
    "LlamaCppServerClient",
    "LlamaCppServerError",
    "LlamaCppVulkanBackend",
]
