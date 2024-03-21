
from configs.agent.gman_config import GmanConfig


class AgentConfig:

    NO_OP = "No_Op"
    GMAN = 'Gman'

    @staticmethod
    def get_config(model_name):

        if model_name == AgentConfig.GMAN:
            return GmanConfig
        else:
            raise ValueError(f"Unknown model {model_name}")
