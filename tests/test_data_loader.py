import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the project root directory to the Python path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_loader import DataLoader

def test_connect_missing_database_file_raises_error():
    """Test that connecting to a non-existent database file raises a FileNotFoundError."""
    invalid_db_path = "non_existent_database_path_12345.duckdb"

    # Ensure the file actually doesn't exist
    assert not os.path.exists(invalid_db_path)

    loader = DataLoader(db_path=invalid_db_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        loader.connect()

    assert f"数据库文件不存在: {invalid_db_path}" in str(excinfo.value)

def test_get_available_tickers_sql_injection_prevention():
    """Test that get_available_tickers parameterizes the limit to prevent SQL injection."""
    loader = DataLoader(db_path="dummy_path.duckdb")

    loader.conn = MagicMock()
    mock_execute = MagicMock()
    mock_execute.fetchall.return_value = []
    mock_execute.fetchone.return_value = [0]
    loader.conn.execute.return_value = mock_execute

    loader.get_available_tickers(limit=5)

    call_args = loader.conn.execute.call_args
    assert call_args is not None

    query, params = call_args[0]

    assert "LIMIT ?" in query
    assert 5 in params
    assert type(params[-1]) is int
