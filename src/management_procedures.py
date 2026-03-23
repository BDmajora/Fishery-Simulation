# src/management_procedures.py
import numpy as np

class TargetBasedManager:
    def __init__(self, Itarget, Etarget, I0=None):
        self.Itarget = float(Itarget)        # target abundance index
        self.Etarget = float(Etarget)        # target fishing effort
        self.I0 = float(I0) if I0 is not None else 0.2 * self.Itarget  # abundance threshold for stronger effort adjustment
        self.recent_indices = []             # store recent observed indices
        self.last_effort = 0.0               # store last applied effort

    def update_abundance_index(self, I_recent):
        self.recent_indices.append(float(I_recent))  # add latest observed abundance

    def get_recent_average(self):
        window = min(4, len(self.recent_indices))   # use up to last 4 observations
        if window == 0:
            return self.Itarget                      # fallback to target if no observations
        return np.mean(self.recent_indices[-window:])  # average of recent indices

    def adjust_effort(self, current_effort, w=0.5, max_change=0.2):
        I_recent = self.get_recent_average()         # recent abundance average
        protection_factor = np.clip(I_recent / self.Itarget, 0.0, 1.0)  # scale effort based on stock recovery

        if len(self.recent_indices) >= 2:
            previous_avg = np.mean(self.recent_indices[-2:-1])
            # spike effort if abundance recovers strongly
            if I_recent > 2 * previous_avg and I_recent > 0.5 * self.Itarget:
                effort_next = self.Etarget * protection_factor
            else:
                if I_recent >= self.I0:
                    # proportional adjustment above threshold
                    effort_next = self.Etarget * protection_factor * (
                        w + (1 - w) * (I_recent - self.I0) / max((self.Itarget - self.I0), 1e-6)
                    )
                else:
                    # quadratic reduction for low abundance
                    effort_next = self.Etarget * w * protection_factor * (I_recent / max(self.I0, 1e-6)) ** 2
        else:
            # conservative adjustment for first steps
            effort_next = self.Etarget * w * protection_factor

        # enforce minimum effort to avoid total shutdown
        min_effort = 0.2 * self.Etarget
        effort_next = max(effort_next, min_effort)

        # limit step-wise changes in effort
        delta = effort_next - current_effort
        max_delta = max(current_effort * max_change, 0.05 * self.Etarget)
        delta = np.clip(delta, -max_delta, max_delta)
        effort_next = current_effort + delta

        # final check for minimum effort
        effort_next = max(effort_next, min_effort)

        self.last_effort = effort_next
        return effort_next
