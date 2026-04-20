from .tam import TAg
def build_temporal_phrase(config):
    if config['temporal_phase'] == True:
        # return MuTRA(**config['mutra'])
        return TAg(config['mutra'])
    else:
        return None