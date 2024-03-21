
import multiprocessing

save_log_lock = multiprocessing.Lock()


network_update_begin_barrier = None
network_update_end_barrier = None


def configure_network_update_barrier(parties, multi_processed):

    global network_update_begin_barrier
    global network_update_end_barrier

    if not multi_processed:
        parties = 1

    network_update_begin_barrier = multiprocessing.Barrier(parties)
    network_update_end_barrier = multiprocessing.Barrier(parties)
