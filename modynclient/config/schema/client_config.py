from pydantic import BaseModel, Field


class Project(BaseModel):
    name: str = Field("Modyn project", description="The name of the project.")
    description: str | None = Field(None, description="The description of the project.")
    version: str | None = Field(None, description="The version of the project.")


class Supervisor(BaseModel):
    ip: str = Field(description="The ip address on which modyn supervisor is run.")
    port: int = Field(description="The port on which modyn supervisor is run.", min=0, max=65535)


class ModynClientConfig(BaseModel):
    """Configuration file for the modyn client.

    It contains the configuration for the client to connect to modyn
    supervisor, adapt as required.
    """

    project: Project = Field(Project(), description="The project information.")
    supervisor: Supervisor = Field(description="The supervisor connection information.")
