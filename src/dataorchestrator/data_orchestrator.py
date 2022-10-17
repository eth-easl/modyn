import pathlib
import json
import sqlite3
import time 

STORAGE_LOCATION = pathlib.Path(__file__).parent.resolve() + '/store'

class DataOrchestrator:
    config = None
    con = None

    def __init__(self, config: dict):
        self.config = config
        self.con = sqlite3.connect(config['data_orchestrator']['in_memory_database'])

    def setup_database(self):
        cur = self.con.cursor()
        # TODO: Add primary key id as integer?
        cur.execute('''CREATE TABLE metadata (
            filename VARCHAR(100) NOT NULL PRIMARY KEY, 
            timestamp INTEGER, 
            score REAL,
            new INTEGER NOT NULL);'''
            )
        self.con.commit()

    def add_batch_to_metadata(self, batch_name: str):
        cur = self.con.cursor()
        cur.execute('''INSERT INTO metadata VALUES(?, ?, ?, ?)''', (batch_name, time.time(), 0, 1))
        self.con.commit()

    def update_metadata(self, batch_name: str, score: float, new: bool):
        cur = self.con.cursor()
        cur.execute('''UPDATE metadata SET score = ?, new = ? WHERE filename = ?''', (score, new, batch_name))
        self.con.commit()

    def update_batches(self):
        # TODO: Read, write and reshuffle data in storage
        pass

    def run(self):
        # TODO: Have a pool of online data loaders ready to feed the trainers
        # TODO: Decide on what data to feed when and where
        pass