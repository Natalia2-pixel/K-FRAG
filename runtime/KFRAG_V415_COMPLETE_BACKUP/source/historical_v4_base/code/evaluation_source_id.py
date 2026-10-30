def evaluation_source_id(
    dataset_name,
    relative_identifier,
):

    message = (
        f"{EVAL_VERSION}|"
        f"{dataset_name}|"
        f"{relative_identifier}"
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        message
    ).digest()[
        :8
    ]
