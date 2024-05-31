from modyn.metadata_database.metadata_base import MetadataBase
from modyn.metadata_database.models.pipelines import Pipeline
from sqlalchemy import Column, ForeignKey, Integer


class AuxiliaryPipeline(MetadataBase):
    """
    AuxiliaryPipeline model.
    """

    __tablename__ = "auxiliary_pipelines"
    pipeline_id = Column("pipeline_id", Integer, ForeignKey(Pipeline.pipeline_id), primary_key=True)
    auxiliary_pipeline_id = Column("auxiliary_pipeline_id", Integer, ForeignKey(Pipeline.pipeline_id), nullable=False)
    __table_args__ = {"extend_existing": True}

    def __repr__(self) -> str:
        """
        Return string representation.
        """
        return f"<AuxiliaryPipeline {self.pipeline_id}, {self.auxiliary_pipeline_id}>"
