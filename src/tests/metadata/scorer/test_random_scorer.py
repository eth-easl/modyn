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

    def test_create_shuffled_training_setes(self):
        filename1 = 'test1.json'
        filename2 = 'test2.json'
        filename3 = 'test3.json'
        filename4 = 'test4.json'

        cursor = self.scorer._con.cursor()

        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.1, 0.3, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.5, 0.8, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (11, 0.7, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (15, 0.5, 1))

        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (10, filename4, 0.5))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             10))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (11, filename4, 0.8))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             11))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (12, filename4, 0.3))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             12))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (20, filename2, 0.9))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (2,
             20))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (21, filename2, 0.7))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (2,
             21))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (22, filename2, 0.1))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (2,
             22))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (30, filename3, 0.1))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (3,
             30))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (31, filename3, 0.2))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (3,
             31))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (32, filename3, 0.8))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (3,
             32))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (40, filename3, 0.1))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (4,
             40))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (41, filename3, 0.2))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (4,
             41))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (42, filename3, 0.8))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (4,
             42))

        self.scorer._create_shuffled_training_setes(2, 4)

        cursor.execute(
            '''SELECT * FROM sample_metadata
            JOIN training_set_to_sample ON training_set_to_sample.sample = sample_metadata.sample
            WHERE training_set_id = 5;''')
        entry = cursor.fetchall()
        self.assertEqual(len(entry), 4)
        expected_results = [20, 21, 31, 32]
        for idx, r in enumerate(entry):
            self.assertIn(r[0], expected_results)

        cursor.execute(
            '''SELECT score FROM sample_metadata
            JOIN training_set_to_sample ON training_set_to_sample.sample = sample_metadata.sample
            WHERE training_set_id = 5;''')
        scores = cursor.fetchall()
        scores = [s[0] for s in scores]
        print(scores)
        cursor.execute(
            '''SELECT score FROM training_set_metadata WHERE id = 5;''')
        training_set_score = cursor.fetchall()[0][0]
        self.assertEqual(
            self.scorer._get_cumulative_score(scores),
            training_set_score)

    def test_add_training_set(self):
        test_file = 'test_file1.csv'
        samples1 = [6, 7, 8]

        self.scorer._add_training_set(test_file, samples1)

        cursor = self.scorer._con.cursor()
        cursor.execute("SELECT * FROM training_set_metadata;")
        sample = cursor.fetchall()[0]
        training_set_id = sample[0]
        cursor.execute(
            '''SELECT * FROM sample_metadata JOIN training_set_to_sample
            ON training_set_to_sample.sample = sample_metadata.sample
            WHERE training_set_id = ?;''',
            (training_set_id,
             ))
        samples = cursor.fetchall()
        self.assertEqual(samples[0][1], test_file)

        cursor.execute("SELECT * FROM sample_metadata ORDER BY sample ASC;")
        result_samples = cursor.fetchall()
        self.assertEqual(result_samples[0][0], samples1[0])
        self.assertTrue(0 <= result_samples[0][2] <= 1)

        self.assertEqual(result_samples[2][0], samples1[2])
        self.assertTrue(0 <= result_samples[2][2] <= 1)

        samples2 = [42, 96, 106]

        self.scorer._add_training_set(test_file, samples2)

        cursor.execute("SELECT * FROM training_set_metadata WHERE id=2;")
        sample = cursor.fetchall()[0]
        training_set_id = sample[0]
        cursor.execute(
            '''SELECT * FROM sample_metadata JOIN training_set_to_sample
            ON training_set_to_sample.sample = sample_metadata.sample
            WHERE training_set_id = ?;''',
            (training_set_id,
             ))
        samples = cursor.fetchall()
        self.assertEqual(samples[0][1], test_file)
        self.assertTrue(0 <= sample[2] <= 1)
        self.assertEqual(sample[3], 1)

        cursor.execute(
            '''SELECT * FROM sample_metadata JOIN training_set_to_sample
            ON training_set_to_sample.sample = sample_metadata.sample
            WHERE training_set_id=2 ORDER BY sample ASC;''')
        result_samples = cursor.fetchall()
        self.assertEqual(result_samples[0][0], samples2[0])
        self.assertTrue(0 <= result_samples[0][2] <= 1)

        self.assertEqual(result_samples[2][0], samples2[2])
        self.assertTrue(0 <= result_samples[2][2] <= 1)

    def test_get_cumulative_score(self):
        result = self.scorer._get_cumulative_score([1, 0.5, 0.5, 0])
        self.assertEqual(result, 0.5)
