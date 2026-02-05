## Last Meeting's Tasks
- Refactoring / rewrite of `generate_decoder_branch`
- Decoupled with AE, use AE model for first stage, original model for second stage $\checkmark$
	- `StopGradient`?
- Validation sets having sample weights (handle all possible combinations?) $\checkmark$
- Add check for $n' \neq n$ (sum of provided weights does not equal number of samples, in fit functions) $\checkmark$
- Fixed bug with metrics being duplicated (this was an UNTRACABLE bug. Thanks Python.) $\checkmark$
# 2/2/26
## Tasks:
- `EarlyStopping` tables (see below)
- For validation, only necessary to support the same data types as TF $\checkmark$
- Bug with Metrics causing crash for decoupled fit $\checkmark$
- Documentation overhaul (WIP)
- **Later on:** Refactoring / rewrite of `generate_decoder_branch`
## Notes:

---
## Regression on SEP-C

| Method               | Representation Layer | Epochs | Time (s) | Frequent MSE | Rare MSE |
| -------------------- | -------------------- | ------ | -------- | ------------ | -------- |
| ==Regular w/o AE==   | ==N/A==              | 1057   | 66.42    | 0.313        | 17.279   |
| ==Balanced w/o AE==  | ==N/A==              |        |          |              |          |
| ==Decoupled w/o AE== | ==-2==               |        |          |              |          |
| ==Decoupled w/o AE== | ==-3==               |        |          |              |          |
| Regular w/ AE        | -2                   |        |          |              |          |
| Balanced w/ AE       | -2                   |        |          |              |          |
| Decoupled w/ AE      | -2                   |        |          |              |          |
| Regular w/ AE        | -3                   |        |          |              |          |
| Balanced w/ AE       | -3                   |        |          |              |          |
| Decoupled w/ AE      | -3                   |        |          |              |          |
## Classification on SEP-C

| Method               | Representation Layer | Epochs | Time (s) | Rare F1 | Rare AUROC |
| -------------------- | -------------------- | ------ | -------- | ------- | ---------- |
| ==Regular w/o AE==   | ==N/A==              |        |          |         |            |
| ==Balanced w/o AE==  | ==N/A==              |        |          |         |            |
| ==Decoupled w/o AE== | ==-2==               |        |          |         |            |
| ==Decoupled w/o AE== | ==-3==               |        |          |         |            |
| Regular w/ AE        | -2                   |        |          |         |            |
| Balanced w/ AE       | -2                   |        |          |         |            |
| Decoupled w/ AE      | -2                   |        |          |         |            |
| Regular w/ AE        | -3                   |        |          |         |            |
| Balanced w/ AE       | -3                   |        |          |         |            |
| Decoupled w/ AE      | -3                   |        |          |         |            |
## Regression on `AgeDB`

| Method           | Representation Layer | Epochs | Time (s) | Frequent MSE | Rare MSE |
| ---------------- | -------------------- | ------ | -------- | ------------ | -------- |
| Regular w/o AE   | N/A                  |        |          |              |          |
| Balanced w/o AE  | N/A                  |        |          |              |          |
| Decoupled w/o AE | -2                   |        |          |              |          |
| Decoupled w/o AE | -3                   |        |          |              |          |
| Regular w/ AE    | -2                   |        |          |              |          |
| Balanced w/ AE   | -2                   |        |          |              |          |
| Decoupled w/ AE  | -2                   |        |          |              |          |
| Regular w/ AE    | -3                   |        |          |              |          |
| Balanced w/ AE   | -3                   |        |          |              |          |
| Decoupled w/ AE  | -3                   |        |          |              |          |
## Classification on `AgeDB`

| Method           | Representation Layer | Epochs | Time (s) | Rare F1 | Rare AUROC |
| ---------------- | -------------------- | ------ | -------- | ------- | ---------- |
| Regular w/o AE   | N/A                  |        |          |         |            |
| Balanced w/o AE  | N/A                  |        |          |         |            |
| Decoupled w/o AE | -2                   |        |          |         |            |
| Decoupled w/o AE | -3                   |        |          |         |            |
| Regular w/ AE    | -2                   |        |          |         |            |
| Balanced w/ AE   | -2                   |        |          |         |            |
| Decoupled w/ AE  | -2                   |        |          |         |            |
| Regular w/ AE    | -3                   |        |          |         |            |
| Balanced w/ AE   | -3                   |        |          |         |            |
| Decoupled w/ AE  | -3                   |        |          |         |            |

---

## Notes:
- Although issue with re-compilation was fixed (sort of band-aid fix), should recompilation be allowed at all?
	- Do we expect the user to change the optimizer, loss, or metrics?
	- Disallowing this would prevent the need for a "band-aid" fix
