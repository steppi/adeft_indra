from contextlib import closing
import logging
import pandas as pd
import pickle
import sqlite3
from typing import Any, Iterator, List, Tuple


from adeft_indra.locations import RESULTS_DB_PATH

logger = logging.getLogger(__file__)


class ResultsManager:
    """Singleton managing an sqlite database for storing results.

    Results tables function as key, value stores. Keys should be strings
    and values can be anything that is pickle serializable.

    Within a table, only one entry can be stored for a given key at any
    one time. The main use case is situations where there are many jobs
    to be run parametrized by a set of parameters, i.e. for each combination
    of parameters there is one and only one job to run. Each parameter
    combination is encoded into a key for which the results corresponding
    to the parameter combination are stored in the table. In case of error,
    system crash, or termination of a cloud instance, the batch of jobs can
    be restarted without recomputing those for any parameter combinations that
    are already in the results table.
    """
    @classmethod
    def add_table(cls, table: str) -> None:
        """Add a key, value table to the database."""
        table_query = f"""--
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            value BLOB,
            UNIQUE(key)
        );
        """
        index_query = f"""--
        CREATE INDEX IF NOT EXISTS
            {table}_idx
        ON
            {table} (key);
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in table_query, index_query:
                    cur.execute(query)

    @classmethod
    def show_tables(cls) -> List[str]:
        """Get list of tables currently available."""
        query = """--
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type = 'table' AND
            name NOT LIKE 'sqlite_%'
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                result = cur.execute(query).fetchall()
        if not result:
            return []
        return [row[0] for row in result]

    @classmethod
    def insert(cls, table: str, key: str, value: Any) -> None:
        """Insert an entry into a table.

        There can only be one entry per key at any given time. User
        will be warned if they try to insert an entry for an existing
        key.
        """
        assert table in cls.show_tables()
        assert isinstance(key, str)
        if cls.get(table, key) is not None:
            logger.warning(f"Value already inserted for key {key}")
            return
        query = f"""--
        INSERT INTO
            {table} (key, value)
        VALUES
            (?, ?);
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    query,
                    (
                        key,
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                )
            conn.commit()

    @classmethod
    def get(cls, table: str, key: str) -> Any:
        assert isinstance(key, str)
        query = f"""--
        SELECT
            value
        FROM
            {table}
        WHERE
            key = ?;
        """
        assert table in cls.show_tables()
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                result = cur.execute(query, (key, )).fetchone()
        if not result:
            return None
        return pickle.loads(result[0])

    @classmethod
    def remove(cls, table: str, key: str) -> None:
        """Remove entry in table associated to a given key."""
        assert isinstance(key, str)
        assert table in cls.show_tables()
        query = f"""--
        DELETE FROM
            {table}
        WHERE
            key = ?;
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, (key, ))

    @classmethod
    def iterrows(cls, table: str) -> Iterator[Tuple[str, Any]]:
        """Iterate through key, value pairs in a given table."""
        assert table in cls.show_tables()
        query = f"SELECT key, value FROM {table}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for row in cur.execute(query):
                    key, value = row
                    yield key, pickle.loads(value)

    @classmethod
    def get_dataset(cls, table: str) -> pd.DataFrame:
        new_rows = []
        for row in cls.iterrows(table):
            new_rows.append(process_row(row))
        return pd.DataFrame(
            new_rows,
            columns=[
                'shortform',
                'grounding',
                'nu',
                'max_features',
                'num_entrez',
                'num_mesh',
                'sens_neg_set',
                'mean_spec',
                'std_spec',
                'J',
                'N_inlier',
                'K_inlier',
                'N_outlier',
                'K_outlier',
            ]
        )

    @classmethod
    def drop_table(cls, table: str) -> None:
        """Drop a table."""
        query = f"DROP TABLE {table}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query)


def process_row(results_row) -> list:
    key, results_json = results_row
    shortform, remainder = key.split(':', maxsplit=1)
    grounding, _ = remainder.split('[', maxsplit=1)
    num_entrez = results_json['train_info']['num_entrez_texts']
    num_mesh = results_json['train_info']['num_mesh_texts']
    best_params = results_json['best_params']
    best_params = (best_params['nu'], best_params['max_features'])
    best_nu, best_max_features = best_params
    train_stats = results_json['train_stats'][best_params]
    sens_neg_set = train_stats[0]
    mean_spec = train_stats[2]
    std_spec = train_stats[3]
    J = train_stats[4]
    labels = results_json['test_info']['labels']
    preds = results_json['test_info']['preds']
    N_inlier = len([label for label in labels if label == grounding])
    K_inlier = len(
        [
            label for label, pred in zip(labels, preds)
            if label == grounding and pred == 1
        ]
    )
    N_outlier = len([label for label in labels if label != grounding])
    K_outlier = len(
        [
            label for label, pred in zip(labels, preds)
            if label != grounding and pred == -1
        ]
    )
    return [
        shortform,
        grounding,
        best_nu,
        best_max_features,
        num_entrez,
        num_mesh,
        sens_neg_set,
        mean_spec,
        std_spec,
        J,
        N_inlier,
        K_inlier,
        N_outlier,
        K_outlier,
    ]
