# Cloud jobs

`strata-column-sweep.yaml` is an approval-gated portability template, not an
active deployment. It intentionally fails while the cost ceiling is zero. The
first repository goal must run locally.

Before using this template:

1. finish Phase 1;
2. implement and test the Column training/data path locally;
3. replace the example command with the reviewed profile experiment;
4. select a live provider/GPU based on current price and availability;
5. test checkpoint recovery and teardown; and
6. obtain approval for the exact dollar ceiling and data placement.
