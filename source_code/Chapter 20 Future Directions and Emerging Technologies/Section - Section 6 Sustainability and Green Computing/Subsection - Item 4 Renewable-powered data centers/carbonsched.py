def schedule_jobs(jobs,ci_forecast,energy_budget): 
    # jobs: list of (energy_kWh, deadline_slot)
    # ci_forecast: array of CI per slot
    # energy_budget: array of available renewable kWh per slot
    schedule = {}
    for job in sorted(jobs, key=lambda j: j[1]):  # earliest deadline first
        e, dl = job
        # pick slot <= dl with max renewable fraction
        candidates = [(s, min(energy_budget[s], e)) for s in range(dl+1)]
        # score by low CI and available energy
        best = max(candidates, key=lambda x: (energy_budget[x[0]]/ (ci_forecast[x[0]]+1e-6)))
        if best[1] >= e:
            schedule[job] = best[0]
            energy_budget[best[0]] -= e
    return schedule