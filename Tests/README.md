# Tests

Unit and integration tests for the `ssoc_autocoder` package.

## Test Files

| File | Tests |
|------|-------|
| `test_processing.py` | Text processing and HTML extraction |
| `test_converting_json.py` | MCF JSON data extraction |
| `test_get_json.py` | JSON fetching |
| `test_json_to_raw.py` | JSON to raw CSV conversion |
| `test_train.py` | Model training |

## Test Data

- `json_test/Ans/` — Expected output JSON files (7 MCF job IDs)
- `json_test/Sol/` — Serialized solution objects
- `functional_test.json` — Functional test cases
- `integration_test_cases.json` — Integration test cases

## Running

```bash
pytest Tests/
```
