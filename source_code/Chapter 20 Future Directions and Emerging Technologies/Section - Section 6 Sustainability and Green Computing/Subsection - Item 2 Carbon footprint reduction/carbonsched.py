def schedule_jobs(jobs, gci_forecast, gpus):  # gci_forecast: list of (hour, kgCO2/kWh)
    # prioritize latency-sensitive jobs; batch flexible jobs to low-carbon windows
    urgent, flexible = partition(jobs, key=lambda j: j.deadline)  # simple split
    assign_urgent(urgent, gpus)  # place to meet latency SLAs
    low_ci_hours = select_low_ci_hours(gci_forecast, window=4)  # choose 4h window
    for job in flexible:
        slot = find_consolidation_slot(job, low_ci_hours, gpus)
        if slot:
            place_job(job, slot)         # pack onto few GPUs
            park_unused_gpus(gpus)      # power-gate idle cards
        else:
            place_job(job, best_effort(gpus))