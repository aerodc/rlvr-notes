# Results ledger

Every stage produces a number. This file is the point of the repo — the code exists to fill in this table.

Rule: a row goes in only when the number came from a command you can re-run, and the command is written down.

| Date | Stage | Metric | Value | How it was measured |
|---|---|---|---|---|
| _pending_ | S0 | Advantage magnitude on learnable task | | `make toy` |
| _pending_ | S0 | Advantage magnitude on flat-reward task | | `make toy-flat` |
| _pending_ | S1 | Checks with evidence attached | | `make test` |
| _pending_ | S1 | Nondeterministic scores per 100 replays | | `make test` |
| _pending_ | S2 | p50 / p95 / p99 reset latency, 1 client | | `make env` + load script |
| _pending_ | S2 | p99 reset latency, 20 concurrent clients | | `make env` + load script |
| _pending_ | S3 | Pass rate where expected group spread peaks | | `make sweep` |
| _pending_ | S3 | Pass-rate band where expected signal < 10% of peak | | `make sweep` |
| _pending_ | S4 | Hacker score vs true success, string verifier | | `s4_failure` |
| _pending_ | S4 | Hacker score vs true success, state verifier | | `s4_failure` |

## Notes

Anything surprising goes here, in a sentence. Future you will want to know why a number moved.

- _(empty)_
