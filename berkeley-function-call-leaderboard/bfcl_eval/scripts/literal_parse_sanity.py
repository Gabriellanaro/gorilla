from bfcl_eval.model_handler.utils import _try_parse_python_literal


def main():
    sample = "{'answer': '7', 'context': 'abc'}"
    parsed = _try_parse_python_literal(sample)
    assert parsed == {"answer": "7", "context": "abc"}

    bad_sample = "{'answer': '7'"
    parsed_bad = _try_parse_python_literal(bad_sample)
    assert parsed_bad is None

    print("literal_parse_sanity: OK")


if __name__ == "__main__":
    main()
