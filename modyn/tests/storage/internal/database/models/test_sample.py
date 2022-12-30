import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.dataset import Dataset

ENGINE = None
SESSION = None


def setup():
    global ENGINE, SESSION  # pylint: disable=global-statement # noqa: E262
    ENGINE = create_engine('sqlite:///:memory:', echo=True)
    SESSION = sessionmaker(bind=ENGINE)()

    Dataset.metadata.create_all(ENGINE)
    File.metadata.create_all(ENGINE)
    Sample.metadata.create_all(ENGINE)


def teardown():
    SESSION.close()
    ENGINE.dispose()


def test_add_sample():
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type='test',
                      file_wrapper_type='test',
                      description='test',
                      version='test')
    SESSION.add(dataset)
    SESSION.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    SESSION.add(file)
    SESSION.commit()

    sample = Sample(file=file, external_key='test', index=0)
    SESSION.add(sample)
    SESSION.commit()

    assert SESSION.query(Sample).filter(Sample.external_key == 'test').first() is not None
    assert SESSION.query(Sample).filter(Sample.external_key == 'test').first().file == file
    assert SESSION.query(Sample).filter(Sample.external_key == 'test').first().index == 0


def test_update_sample():
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type='test',
                      file_wrapper_type='test',
                      description='test',
                      version='test')
    SESSION.add(dataset)
    SESSION.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    SESSION.add(file)
    SESSION.commit()

    sample = Sample(file=file, external_key='test', index=0)
    SESSION.add(sample)
    SESSION.commit()

    SESSION.query(Sample).filter(Sample.external_key == 'test').update({
        'index': 1
    })

    assert SESSION.query(Sample).filter(Sample.external_key == 'test').first().index == 1
