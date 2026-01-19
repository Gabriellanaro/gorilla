from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI


def main() -> None:
    api = WebSearchAPI()
    results = api.search_engine_query("capital of france", max_results=3, region="wt-wt")
    if isinstance(results, dict) and "error" in results:
        print(f"Error: {results['error']}")
        return

    for item in results[:3]:
        title = item.get("title", "")
        href = item.get("href", "")
        print(f"{title} | {href}")


if __name__ == "__main__":
    main()
