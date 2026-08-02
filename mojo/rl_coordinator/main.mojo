"""Unconfigured native boundary for the gepa-actor-v1 coordinator protocol."""


fn is_canonical_nonce(value: StringSlice) -> Bool:
    if len(value) != 32:
        return False
    for index in range(len(value)):
        if not (value[byte=index] in "0123456789abcdef"):
            return False
    return True


def main() raises:
    # The unconfigured coordinator accepts only canonical envelopes and echoes each caller-owned
    # correlation nonce. Errors never include that nonce.
    var hello = input()
    var hello_prefix = (
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"hello\","
        "\"request_id\":\"request-1:"
    )
    var hello_suffix = "\",\"payload\":{}}"
    var hello_nonce_start = len(hello_prefix)
    var hello_nonce_end = hello_nonce_start + 32
    if len(hello) != hello_nonce_end + len(hello_suffix):
        raise Error("invalid gepa-actor-v1 hello frame")
    if not hello.startswith(hello_prefix):
        raise Error("invalid gepa-actor-v1 hello frame")
    var hello_nonce = hello[byte=hello_nonce_start:hello_nonce_end]
    if not is_canonical_nonce(hello_nonce) or hello[byte=hello_nonce_end:] != hello_suffix:
        raise Error("invalid gepa-actor-v1 hello frame")

    print(
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"hello\","
        "\"request_id\":\"request-1:",
        hello_nonce,
        "\",\"payload\":{\"backend_name\":\"mojo-coordinator\","
        "\"backend_version\":\"unconfigured\"}}",
        sep="",
        flush=True,
    )

    # Until a native actor adapter is supplied, request 2 is either a clean close or an explicit
    # unconfigured generate error. This boundary never fabricates a trajectory or trains weights.
    var request = input()
    var close_prefix = (
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"close\","
        "\"request_id\":\"request-2:"
    )
    var close_suffix = "\",\"payload\":{}}"
    var generate_prefix = (
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"generate\","
        "\"request_id\":\"request-2:"
    )
    var generate_suffix = (
        "\",\"payload\":{\"requests\":[{\"case_id\":null,\"metadata\":{},"
        "\"num_samples\":1,\"policy_version\":\"1\",\"prompt\":\"probe\","
        "\"sampling_parameters\":{},\"seed\":null}]}}"
    )
    var close_nonce_start = len(close_prefix)
    var close_nonce_end = close_nonce_start + 32

    if request.startswith(close_prefix) and len(request) == close_nonce_end + len(close_suffix):
        var close_nonce = request[byte=close_nonce_start:close_nonce_end]
        if is_canonical_nonce(close_nonce) and request[byte=close_nonce_end:] == close_suffix:
            print(
                "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"close\","
                "\"request_id\":\"request-2:",
                close_nonce,
                "\",\"payload\":{}}",
                sep="",
                flush=True,
            )
            return

    if not request.startswith(generate_prefix):
        raise Error("invalid gepa-actor-v1 generate or close frame")
    var generate_nonce_start = len(generate_prefix)
    var generate_nonce_end = generate_nonce_start + 32
    if len(request) != generate_nonce_end + len(generate_suffix):
        raise Error("invalid gepa-actor-v1 generate or close frame")
    var generate_nonce = request[byte=generate_nonce_start:generate_nonce_end]
    if not is_canonical_nonce(generate_nonce):
        raise Error("invalid gepa-actor-v1 generate or close frame")
    if request[byte=generate_nonce_end:] != generate_suffix:
        raise Error("invalid gepa-actor-v1 generate or close frame")

    print(
        "{\"protocol_version\":\"gepa-actor-v1\",\"type\":\"error\","
        "\"request_id\":\"request-2:",
        generate_nonce,
        "\",\"payload\":{\"code\":\"actor_unconfigured\","
        "\"message\":\"native actor adapter is not configured\"}}",
        sep="",
        flush=True,
    )
