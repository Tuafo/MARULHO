"""Extended stability test for Terminus — 1hr multimodal run.

Tracks: tick rate, throughput, memory fill, multimodal episodes,
neuromodulators, concept count, cross-modal acceptance, errors.
"""
import requests, time, json, sys
from datetime import datetime

BASE = "http://127.0.0.1:8099"
RUN_SECONDS = 3600  # 1 hour
POLL_INTERVAL = 15  # seconds between polls

print(f"=== Terminus Extended Evaluation ({RUN_SECONDS}s) ===")
print(f"Start: {datetime.now().isoformat()}")

# Quick-start multimodal
r = requests.post(f"{BASE}/terminus/quick-start?preset=multimodal", timeout=60)
assert r.status_code == 200, f"Quick-start failed: {r.status_code}"
print("Multimodal preset started.")

# Tracking arrays
data_points = []
start_time = time.monotonic()
last_tick = 0
last_tokens = 0
tick_times = []
errors = []

try:
    while (time.monotonic() - start_time) < RUN_SECONDS:
        elapsed = time.monotonic() - start_time
        time.sleep(POLL_INTERVAL)

        try:
            # Terminus status
            r = requests.get(f"{BASE}/terminus", timeout=5)
            ts = r.json()
            rt = ts.get("terminus_runtime", {})
            mm = rt.get("multimodal") or {}

            # Full status for neuromod + memory + concepts
            r2 = requests.get(f"{BASE}/status", timeout=5)
            st = r2.json()

            point = {
                "t": round(elapsed, 1),
                "tick": rt.get("tick_count", 0),
                "tokens": rt.get("background_tokens_processed", 0),
                "mm_eps": mm.get("episodes_completed", 0),
                "vis_acc": mm.get("cross_modal_visual_accepted", 0),
                "aud_acc": mm.get("cross_modal_audio_accepted", 0),
                "error": rt.get("last_error"),
                "fill_frac": st.get("memory_store", {}).get("fill_fraction", 0),
                "n_concepts": st.get("concept_store", {}).get("concept_count", 0),
                "dopamine": st.get("dopamine", 0),
                "serotonin": st.get("serotonin", 0),
                "running": rt.get("running", False),
            }
            data_points.append(point)

            # Calculate rates
            new_tick = point["tick"]
            new_tokens = point["tokens"]
            tick_delta = new_tick - last_tick
            token_delta = new_tokens - last_tokens
            tok_per_s = token_delta / POLL_INTERVAL if POLL_INTERVAL > 0 else 0

            status_line = (
                f"[{elapsed:>6.0f}s] tick={new_tick:>3d} "
                f"tok={new_tokens:>6d} ({tok_per_s:>5.1f}/s) "
                f"mm={point['mm_eps']:>3d}({point['vis_acc']}/{point['aud_acc']}) "
                f"fill={point['fill_frac']:.2f} "
                f"C={point['n_concepts']:>4d} "
                f"DA={point['dopamine']:.3f} 5HT={point['serotonin']:.3f}"
            )

            if point["error"]:
                status_line += f" ERR={point['error']}"
                errors.append((elapsed, point["error"]))

            if not point["running"]:
                status_line += " STOPPED!"
                print(status_line, flush=True)
                print("Brain loop stopped unexpectedly!")
                break

            print(status_line, flush=True)
            last_tick = new_tick
            last_tokens = new_tokens

        except requests.exceptions.RequestException as e:
            print(f"[{elapsed:>6.0f}s] POLL ERROR: {e}", flush=True)
            errors.append((elapsed, str(e)))

except KeyboardInterrupt:
    print("\n--- Interrupted ---")

# Summary
elapsed_total = time.monotonic() - start_time
print(f"\n=== Summary ({elapsed_total:.0f}s) ===")
if data_points:
    last = data_points[-1]
    first = data_points[0]
    total_tokens = last["tokens"]
    total_ticks = last["tick"]
    total_eps = last["mm_eps"]
    avg_tok_s = total_tokens / elapsed_total if elapsed_total > 0 else 0
    avg_tick_s = elapsed_total / total_ticks if total_ticks > 0 else 0

    print(f"Total tokens:   {total_tokens}")
    print(f"Total ticks:    {total_ticks}")
    print(f"MM episodes:    {total_eps}")
    print(f"Avg throughput: {avg_tok_s:.1f} tok/s")
    print(f"Avg tick time:  {avg_tick_s:.1f}s")
    print(f"Memory fill:    {last['fill_frac']:.2f}")
    print(f"Concepts:       {last['n_concepts']}")
    print(f"Errors:         {len(errors)}")
    if errors:
        for t, e in errors[:5]:
            print(f"  [{t:.0f}s] {e}")

# Stop
try:
    requests.post(f"{BASE}/terminus/stop", timeout=10)
    print("Terminus stopped.")
except Exception:
    pass

print(f"End: {datetime.now().isoformat()}")
