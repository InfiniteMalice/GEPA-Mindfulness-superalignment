"""Unconfigured native boundary for the gepa-actor-v1 coordinator protocol."""


def main() raises:
    # The unconfigured coordinator accepts only the canonical first hello frame. The Python
    # transport performs the general JSON validation and never sends alternate field ordering.
    var hello = input()
    var expected_hello = (
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"hello\","
        "\"request_id\":\"request-1\",\"payload\":{}}"
    )
    if hello != expected_hello:
        raise Error("invalid gepa-actor-v1 hello frame")

    print(
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"hello\","
        "\"request_id\":\"request-1\",\"payload\":{\"backend_name\":"
        "\"mojo-coordinator\",\"backend_version\":\"unconfigured\"}}",
        sep="",
        flush=True,
    )

    # Until a native actor adapter is supplied, request 2 is either a clean close or an explicit
    # unconfigured generate error. This boundary never fabricates a trajectory or trains weights.
    var request = input()
    var expected_close = (
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"close\","
        "\"request_id\":\"request-2\",\"payload\":{}}"
    )
    if request == expected_close:
        print(
            "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"close\","
            "\"request_id\":\"request-2\",\"payload\":{}}",
            sep="",
            flush=True,
        )
        return

    var expected_generate = (
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"generate\","
        "\"request_id\":\"request-2\",\"payload\":{\"requests\":[{"
        "\"case_id\":null,\"metadata\":{},\"num_samples\":1,"
        "\"policy_version\":\"1\",\"prompt\":\"probe\","
        "\"sampling_parameters\":{},\"seed\":null}]}}"
    )
    if request != expected_generate:
        raise Error("invalid gepa-actor-v1 generate or close frame")

    print(
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"error\","
        "\"request_id\":\"request-2\",\"payload\":{\"code\":"
        "\"actor_unconfigured\",\"message\":\"native actor adapter is not configured\"}}",
        sep="",
        flush=True,
    )
