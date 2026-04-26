When dealing with imbalanced data with a significantly larger number of negative data points than positive ones, the [[False Positive Rate]] should be very close to zero, meaning the [[True Skill Statistic]] close to the [[True Positive Rate]]. At the same time, when the number of false positives is significant, [[True Skill Statistic]] may not be the best metric to use

[[F1]] is the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of [[Precision]] and [[True Positive Rate|Recall]], since the False Positives and False Negatives are similarly stressed with respect to the True Positives. Therefore, when the number of false positives is significant, [[F1]] becomes a more desirable statistic than [[True Skill Statistic]].

For a reasonable system, the number of True Negatives is much larger than the number of False Positives, True Positives, and False Negatives.

[[Heikde Skill Score]] discounts [[Expected True Positive]] and [[Expected True Negative]]
[[Gilbert Skill Score]] does not discount [[Expected True Negative]], but discounts [[Expected True Positive]]
[[Critical Success Index]] does not discount either of these expected values.

[[F1]] and [[Critical Success Index]] are quite similar, the only difference being that the True Positives are multiplied by 2 in [[F1]], but not [[Critical Success Index]].