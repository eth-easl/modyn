# pylint: disable=redefined-outer-name
import json

import pytest
from modyn.metadata_database.models import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Pipeline.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_pipeline(session):
    pipeline = Pipeline(
        num_workers=10,
        model_class_name="ResNet18",
        model_config=json.dumps({"num_classes": 10}),
        amp=True,
        selection_strategy="{}",
        full_model_strategy_name="PyTorchFullModel",
    )
    session.add(pipeline)
    session.commit()

    extracted_pipeline: Pipeline = session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first()

    assert extracted_pipeline is not None
    assert extracted_pipeline.num_workers == 10
    assert extracted_pipeline.model_class_name == "ResNet18"
    assert json.loads(extracted_pipeline.model_config)["num_classes"] == 10
    assert extracted_pipeline.amp
    assert extracted_pipeline.full_model_strategy_name == "PyTorchFullModel"
    assert not extracted_pipeline.full_model_strategy_zip
    assert extracted_pipeline.inc_model_strategy_name is None
    assert extracted_pipeline.full_model_strategy_config is None


def test_update_pipeline(session):
    pipeline = Pipeline(
        num_workers=10,
        model_class_name="ResNet18",
        model_config="{}",
        amp=True,
        selection_strategy="{}",
        full_model_strategy_name="PyTorchFullModel",
    )
    session.add(pipeline)
    session.commit()

    pipeline.num_workers = 20
    pipeline.amp = False
    session.commit()

    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first() is not None
    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first().num_workers == 20
    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first().selection_strategy == "{}"
    assert not session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first().amp

    pipeline.model_class_name = "test_model"
    session.commit()

    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first().model_class_name == "test_model"


def test_delete_pipeline(session):
    pipeline = Pipeline(
        num_workers=10,
        model_class_name="ResNet18",
        model_config="{}",
        amp=False,
        selection_strategy="{}",
        full_model_strategy_name="PyTorchFullModel",
    )
    session.add(pipeline)
    session.commit()

    session.delete(pipeline)
    session.commit()

    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first() is None
