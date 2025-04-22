from fortyfives.envs.fortyfives_env import FortyfivesEnv

# Register the environment
from rlcard.envs import register
register('fortyfives', 'fortyfives.envs.fortyfives_env:FortyfivesEnv')
