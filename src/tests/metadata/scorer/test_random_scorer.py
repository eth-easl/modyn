import pathlib
from mock import patch

from dynamicdatasets.metadata.scorer import RandomScorer
from ..test_metadata import TestMetadata, __metadata_init__

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


class TestRandomScorer(TestMetadata):

    def setUp(self):
        super().setUp()
        with patch.object(RandomScorer, '__init__', __metadata_init__):
            self.scorer = RandomScorer(None)
            self.scorer._setup_database()

    def test_create_shuffled_batches(self):
        filename1 = 'test1.json'
        filename2 = 'test2.json'
        filename3 = 'test3.json'
        filename4 = 'test4.json'

        cursor = self.scorer._con.cursor()

        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.1, 0.3, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.5, 0.8, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (11, 0.7, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (15, 0.5, 1))

        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (10, filename4, 0.5))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 10))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (11, filename4, 0.8))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 11))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (12, filename4, 0.3))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 12))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (20, filename2, 0.9))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (2, 20))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (21, filename2, 0.7))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (2, 21))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (22, filename2, 0.1))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (2, 22))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (30, filename3, 0.1))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (3, 30))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (31, filename3, 0.2))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (3, 31))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (32, filename3, 0.8))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (3, 32))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (40, filename3, 0.1))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (4, 40))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (41, filename3, 0.2))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (4, 41))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (42, filename3, 0.8))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (4, 42))

        self.scorer._create_shuffled_batches(2, 4)

        cursor.execute(
            '''SELECT * FROM row_metadata
            JOIN batch_to_row ON batch_to_row.row = row_metadata.row WHERE batch_id = 5;''')
        entry = cursor.fetchall()
        self.assertEqual(len(entry), 4)
        expected_results = [20, 21, 31, 32]
        for idx, r in enumerate(entry):
            self.assertIn(r[0], expected_results)

        cursor.execute(
            '''SELECT score FROM row_metadata
            JOIN batch_to_row ON batch_to_row.row = row_metadata.row WHERE batch_id = 5;''')
        scores = cursor.fetchall()
        scores = [s[0] for s in scores]
        print(scores)
        cursor.execute('''SELECT score FROM batch_metadata WHERE id = 5;''')
        batch_score = cursor.fetchall()[0][0]
        self.assertEqual(
            self.scorer._get_cumulative_score(scores),
            batch_score)

    def test_add_batch(self):
        test_file = 'test_file1.csv'
        rows1 = [6, 7, 8]

        self.scorer._add_batch(test_file, rows1)

        cursor = self.scorer._con.cursor()
        cursor.execute("SELECT * FROM batch_metadata;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        cursor.execute(
            "SELECT * FROM row_metadata JOIN batch_to_row ON batch_to_row.row = row_metadata.row WHERE batch_id = ?;",
            (batch_id,
             ))
        rows = cursor.fetchall()
        self.assertEqual(rows[0][1], test_file)

        cursor.execute("SELECT * FROM row_metadata ORDER BY row ASC;")
        result_rows = cursor.fetchall()
        self.assertEqual(result_rows[0][0], rows1[0])
        self.assertTrue(0 <= result_rows[0][2] <= 1)

        self.assertEqual(result_rows[2][0], rows1[2])
        self.assertTrue(0 <= result_rows[2][2] <= 1)

        rows2 = [42, 96, 106]

        self.scorer._add_batch(test_file, rows2)

        cursor.execute("SELECT * FROM batch_metadata WHERE id=2;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        cursor.execute(
            "SELECT * FROM row_metadata JOIN batch_to_row ON batch_to_row.row = row_metadata.row WHERE batch_id = ?;",
            (batch_id,
             ))
        rows = cursor.fetchall()
        self.assertEqual(rows[0][1], test_file)
        self.assertTrue(0 <= row[2] <= 1)
        self.assertEqual(row[3], 1)

        cursor.execute(
            '''SELECT * FROM row_metadata JOIN batch_to_row ON batch_to_row.row = row_metadata.row
            WHERE batch_id=2 ORDER BY row ASC;''')
        result_rows = cursor.fetchall()
        self.assertEqual(result_rows[0][0], rows2[0])
        self.assertTrue(0 <= result_rows[0][2] <= 1)

        self.assertEqual(result_rows[2][0], rows2[2])
        self.assertTrue(0 <= result_rows[2][2] <= 1)

    def test_get_cumulative_score(self):
        result = self.scorer._get_cumulative_score([1, 0.5, 0.5, 0])
        self.assertEqual(result, 0.5)
