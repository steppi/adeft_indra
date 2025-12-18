import pickle
import sqlite3

from contextlib import closing
from typing import Any, Iterator, Tuple


class DuplicateKeyError(ValueError):
    """Raised when attempting to update the value associated to an existing key.

    """

class ResultsManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

        query = f"""--
        CREATE TABLE IF NOT EXISTS results_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value BLOB
        );
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute(query)
            conn.commit()

    def _validate_key(self, key):
        if not isinstance(key, str):
            raise TypeError(
                "ResultsManager supports only string valued keys, "
                f"received {key}"
            )
        return key

    def __getitem__(self, key):
        key = self._validate_key(key)
        query = f"""--
        SELECT
            value
        FROM
            results_table
        WHERE
            key = ?;
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                result = cur.execute(query, (key, )).fetchone()
        if not result:
            raise KeyError
        return pickle.loads(result[0])

    def __contains__(self, key):
        key = self._validate_key(key)
        query = """--
        SELECT
            value
        FROM
            results_table
        WHERE
            key = ?;
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
             with closing(conn.cursor()) as cur:
                 result = cur.execute(query, (key, )).fetchone()
        return bool(result)

    def __setitem__(self, key, value):
        key = self._validate_key(key)
        query = f"""--
        INSERT INTO
            results_table (key, value)
        VALUES
            (?, ?);
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                try:
                    cur.execute(
                        query,
                        (
                            key,
                            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                        )
                    )
                except sqlite3.IntegrityError:
                    raise DuplicateKeyError(
                        f"Attempted to update value for existing key {key}."
                    )
            conn.commit()

    def __delitem__(self, key):
        key = self._validate_key(key)
        query = """--
        DELETE FROM results_table
        WHERE key = ?;
        """

        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, (key,))
                if cur.rowcount == 0:
                    # no row deleted means key didn't exist
                    raise KeyError(key)
            conn.commit()
        
    def get(self, key, default=None):
        key = self._validate_key(key)
        if key in self:
            return self[key]
        return default

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Iterate through key, value pairs."""
        query = f"SELECT key, value FROM results_table"
        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                for key, raw_value in cur.execute(query):
                    yield key, pickle.loads(raw_value)

    def keys(self) -> Iterator[str]:
        """Iterate through keys."""
        query = f"SELECT key FROM results_table"
        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                for (key,) in cur.execute(query):
                    yield key

    def values(self) -> Iterator[Any]:
        """Iterate through values."""
        query = f"SELECT value FROM results_table"
        with closing(sqlite3.connect(self.db_path)) as conn:
            with closing(conn.cursor()) as cur:
                for (value,) in cur.execute(query):
                    yield value
