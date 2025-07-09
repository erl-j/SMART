import symusic

def sm_beats_per_second(sm):
    # get how many seconds are in a beat
    return 60 / sm.tempos[-1].qpm

def sm_seconds(sm):
    # get how many seconds are in the whole midi file
    sm_copy = sm.copy()
    # convert to seconds
    sm_copy = sm_copy.to(symusic.TimeUnit.second)
    return sm_copy.end()

def crop_sm(sm, n_beats):
    """
    Crop a symbolic music object to a specific number of beats.
    
    Parameters:
    -----------
    sm : object
        Symbolic music object with tpq attribute and clip method
    n_beats : int
        Number of beats to keep
        
    Returns:
    --------
    object
        Cropped symbolic music object
    """
    # Create a copy to avoid modifying the original
    sm_copy = sm.copy()
    tpq = sm_copy.tpq

    # first check that the end is not less than n_beats
    if sm_copy.end() > n_beats * tpq:
        # Clip to specified number of beats
        sm_copy = sm_copy.clip(0, n_beats * tpq, clip_end=True)
    
    return sm_copy
