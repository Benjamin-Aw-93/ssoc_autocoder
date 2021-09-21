import pytest
from ssoc_autocoder.processing import *


class TestProcessing:

    def test_remove_prefix(self):
        assert remove_prefix("123 hello", ['123']) == "hello"


if __name__ == '__main__':
    pytest.main()
