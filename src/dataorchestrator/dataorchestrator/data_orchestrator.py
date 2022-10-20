import pathlib
import json
import sqlite3
import time 

class DataOrchestrator:
    config = None
    con = None

    def __init__(self, config: dict):
        self.config = config
        self.con = sqlite3.connect(config['data_orchestrator']['in_memory_database'])
        self.setup_database()

    def setup_database(self):
        cur = self.con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            filename VARCHAR(100), 
            timestamp INTEGER, 
            score REAL,
            new INTEGER NOT NULL);'''
            )
        self.con.commit()

    def add_batch_to_metadata(self, filename: str):
        cur = self.con.cursor()
        cur.execute('''INSERT INTO metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''', (filename, time.time(), 0, 1))
        self.con.commit()

    def update_metadata(self, batch_id: int, score: float, new: bool):
        cur = self.con.cursor()
        cur.execute('''UPDATE metadata SET score = ?, new = ? WHERE id = ?''', (score, new, batch_id))
        self.con.commit()

    def update_batches(self):
        # TODO: Read, write and reshuffle data in storage
        pass

    def run(self):
        # TODO: Have a pool of online data loaders ready to feed the trainers
        # TODO: Decide on what data to feed when and where
        pass