import contextlib
import logging
import os
import signal
import threading
import time

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def shared_init_stage(
    lock_dir: str,
    poll_interval_s: float = 15.0,
    stale_after_s: float = 2 * 60 * 60,
    heartbeat_interval_s: float = 60.0,
):
    """Coordinate a pipeline stage whose outputs are shared across concurrently
    launched runs (e.g. parallel `conformal_policy_control.alpha` sweeps for the
    same `run_name`/seed, which produce identical GA data and initial-SFT
    checkpoints).

    The first process to reach this point becomes the "leader": it claims the
    stage via an atomic file creation, periodically "heartbeats" the claim while
    it works, and runs the wrapped code as normal. Any other process that reaches
    this point while the leader is still working becomes a "follower": it blocks
    here (polling for the leader's completion marker) instead of redoing the same
    work, then runs the wrapped code itself — which is safe because the stage's
    own `overwrite_*=False` / file-exists checks are unchanged, so the follower
    just finds the artifacts already on disk and skips straight through.

    If the leader dies before finishing (crash, OOM-kill, hard node failure,
    ...) with no chance to clean up, its heartbeat goes stale. Once a follower
    observes no heartbeat for more than `stale_after_s`, it atomically steals
    the claim and becomes the new leader, so the pipeline doesn't hang forever
    waiting on a process that's gone. Staleness is judged by heartbeat recency
    rather than total elapsed time, so a leader that's simply running long
    (queueing, slow disk, ...) never gets its claim stolen out from under it.

    Two failure modes are handled faster than waiting out `stale_after_s`:
    - A `SIGTERM` (e.g. a Slurm preemption or cancellation) makes the leader
      release its claim immediately before exiting, so a follower can take
      over right away instead of waiting out the full staleness timeout.
    - If the wrapped code raises an exception, the claim is released (not
      marked done) before the exception propagates, so a genuinely failed
      init stage is retried rather than being mistaken for a successful one.
    """
    os.makedirs(lock_dir, exist_ok=True)
    claim_fp = os.path.join(lock_dir, "leader.claim")
    done_fp = os.path.join(lock_dir, "leader.done")

    while True:
        if os.path.exists(done_fp):
            yield
            return

        try:
            fd = os.open(claim_fp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            pass
        else:
            break  # we created it: we're the leader

        logger.info(
            f"Another run already claimed the shared init stage ({claim_fp}); "
            f"waiting for it to finish (marker: {done_fp})..."
        )
        if _wait_for_done_or_steal(claim_fp, done_fp, poll_interval_s, stale_after_s):
            logger.warning(
                f"No heartbeat on {claim_fp} for over {stale_after_s / 60:.0f} min; "
                "assuming the previous leader died and taking over the shared init stage."
            )
            break
        # Otherwise done_fp appeared (or the claim vanished because another
        # follower already stole it) -- loop back to the top to recheck.

    logger.info(f"Claimed shared init stage ({claim_fp}); running it now.")
    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(claim_fp, heartbeat_interval_s, stop_heartbeat),
        daemon=True,
    )
    heartbeat_thread.start()

    def _release_claim_on_terminate(signum, frame):
        logger.warning(
            f"Received signal {signum} while leading the shared init stage; "
            f"releasing claim ({claim_fp}) so another run can take over, "
            "then exiting."
        )
        stop_heartbeat.set()
        try:
            os.remove(claim_fp)
        except FileNotFoundError:
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    prev_handler = signal.signal(signal.SIGTERM, _release_claim_on_terminate)
    try:
        yield
    except BaseException:
        logger.warning(
            f"Shared init stage raised an exception; releasing claim "
            f"({claim_fp}) so another run can retry it, without marking it done."
        )
        stop_heartbeat.set()
        heartbeat_thread.join()
        try:
            os.remove(claim_fp)
        except FileNotFoundError:
            pass
        raise
    else:
        stop_heartbeat.set()
        heartbeat_thread.join()
        with open(done_fp, "w"):
            pass
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


def _heartbeat_loop(claim_fp: str, interval_s: float, stop_event: threading.Event):
    while not stop_event.wait(interval_s):
        try:
            os.utime(claim_fp, None)
        except FileNotFoundError:
            return


def _wait_for_done_or_steal(
    claim_fp: str, done_fp: str, poll_interval_s: float, stale_after_s: float
) -> bool:
    """Poll until the leader finishes, or steal the claim once its heartbeat
    goes stale.

    Returns True if this process stole the claim and should become the new
    leader; False if `done_fp` appeared, or if stealing was attempted but lost
    to a race with another follower (in which case the caller loops back
    around to find out who is leader now).
    """
    waited_s = 0.0
    while True:
        if os.path.exists(done_fp):
            return False
        time.sleep(poll_interval_s)
        waited_s += poll_interval_s
        if waited_s % (poll_interval_s * 20) < poll_interval_s:
            logger.info(
                f"Still waiting on shared init stage ({done_fp}); "
                f"{waited_s / 60:.1f} min elapsed."
            )

        try:
            heartbeat_age_s = time.time() - os.path.getmtime(claim_fp)
        except FileNotFoundError:
            # Claim vanished without a done marker: another follower already
            # stole it (or is in the process of doing so).
            return False

        if heartbeat_age_s < stale_after_s:
            continue

        # Heartbeat is stale: try to steal the claim. Only the one process
        # whose `remove` call actually succeeds proceeds to recreate it, so
        # concurrent stealers can never both become leader.
        try:
            os.remove(claim_fp)
        except FileNotFoundError:
            return False  # someone else already stole it (or leader finished)

        try:
            fd = os.open(claim_fp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            # Vanishingly unlikely: a fresh claim from a brand-new run raced
            # in between our remove() and open(). Defer to it instead.
            return False

        return True
