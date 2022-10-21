import pathlib
import json
import time
import sqlite3
import random
import statistics


class DataOrchestrator:
    config = None
    con = None

    def __init__(self, config: dict):
        self.config = config
        self.con = sqlite3.connect(
            config['data_orchestrator']['in_memory_database'])
        self.setup_database()

    def setup_database(self):
        cur = self.con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS file_metadata (
            id INTEGER PRIMARY KEY,
            filename VARCHAR(100), 
            timestamp INTEGER, 
            score REAL,
            new INTEGER NOT NULL);'''
                    )
        cur.execute('''CREATE TABLE IF NOT EXISTS row_metadata (
            row INTEGER,
            file_id INTEGER, 
            score REAL,
            PRIMARY KEY (row, file_id),
            FOREIGN KEY (file_id) REFERENCES file_metadata(id));'''
                    )
        self.con.commit()

    def add_file(self, filename: str, rows: list[int]):
        file_id = self.add_file_to_metadata(filename)
        scores = []
        for row in rows:
            score = self.get_initial_random_score()
            self.add_row_to_metadata(row, file_id, score)
            scores.append(score)
        median = statistics.median(scores)
        self.update_file_metadata(file_id, median, 1)

    def add_file_to_metadata(self, filename: str):
        cur = self.con.cursor()
        cur.execute('''INSERT INTO file_metadata(filename, timestamp, new) VALUES(?, ?, ?)''',
                    (filename, time.time(), 1))
        file_id = cur.lastrowid
        self.con.commit()
        return file_id

    def add_row_to_metadata(self, row: int, file_id: int, score: float):
        cur = self.con.cursor()
        cur.execute('''INSERT INTO row_metadata(row, file_id, score) VALUES(?, ?, ?)''',
                    (row, file_id, score))
        self.con.commit()

    def update_file_metadata(self, file_id: int, score: float, new: bool):
        cur = self.con.cursor()
        cur.execute(
            '''UPDATE file_metadata SET score = ?, new = ? WHERE id = ?''', (score, new, file_id))
        self.con.commit()

    def update_row_metadata(self, file_id: int, row: int, score: float):
        cur = self.con.cursor()
        cur.execute(
            '''UPDATE row_metadata SET score = ? WHERE file_id = ? AND row = ?''', (score, file_id, row))
        self.con.commit()

    def get_initial_random_score(self):
        return random.uniform(0, 1)

    def update_batches(self):
        # TODO: Read, write and reshuffle data in storage
        pass

    def run(self):
        # TODO: Have a pool of online data loaders ready to feed the trainers
        # TODO: Decide on what data to feed when and where
        pass

    def prune(self):
        # TODO: Think about how to delete data if necessary
        pass
