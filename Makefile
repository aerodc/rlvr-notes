.PHONY: help toy toy-flat toy-hard toy-noisy test env env-load env-test sweep clean

help:
	@echo "S0  make toy | toy-flat | toy-hard | toy-noisy   (no GPU, no deps)"
	@echo "S1  make test                                    verifier tests"
	@echo "S2  make env                                     start env service on :8787"
	@echo "    make env-test                                end-to-end client check"
	@echo "    make env-load                                reset latency under concurrency"
	@echo "S3  make sweep                                   difficulty calibration sweep"

toy:
	python3 s0_baseline/toy_grpo.py --task learnable

toy-flat:
	python3 s0_baseline/toy_grpo.py --task flat

toy-hard:
	python3 s0_baseline/toy_grpo.py --task hard

toy-noisy:
	python3 s0_baseline/toy_grpo.py --task noisy

test:
	python3 -m pytest s1_verifier -q

env:
	node s2_env_service/server.mjs

env-test:
	python3 s2_env_service/client.py --selftest

env-load:
	python3 s2_env_service/client.py --load

sweep:
	python3 s3_difficulty/sweep.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache runs
