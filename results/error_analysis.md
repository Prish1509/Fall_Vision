# Error analysis (Step 25)

12 lowest-scoring missed falls (OOPS test) and 6 highest-scoring false alarms, watched by hand.

Checked against the label files: 9 of the 12 missed falls scored exactly 0.0 because the person
is detected in only 6-21 % of frames, so no 2-s window was ever scored (a detection failure,
not a classifier failure). 77 of 818 OOPS clips are in the falls/ folder but have no 'fall'
label, so some OOPS 'false alarms' are label artefacts.

## Missed falls

| Clip | Reason | Note |
|---|---|---|
| missed_01 | small | person tiny or far away |
| missed_02 | small | person tiny or far away |
| missed_03 | camera | camera shakes, pans or zooms a lot |
| missed_04 | fast | fall too quick or blurry |
| missed_05 | camera | camera shakes, pans or zooms a lot |
| missed_06 | offframe | person partly or fully out of the frame |
| missed_07 | other | person falls backward while sitting |
| missed_08 | small | person tiny or far away |
| missed_09 | small | person tiny or far away |
| missed_10 | camera | camera shakes, pans or zooms a lot |
| missed_11 | other | person falls while trying to sit on a horse |
| missed_12 | fast | fall too quick or blurry |

## False alarms

| Clip | Reason | Note |
|---|---|---|
| false_01 | fast | fall too quick |
| false_02 | label | GMDCSA24 S2 Fall 03 is labelled lie_down + lying, no fall segment |
| false_03 | lying | person lies down on purpose (false alarm) |
| false_04 | notperson | the “fall” is a bike, skateboard, trampoline or object |
| false_05 | lying | person lies down on purpose (false alarm) |
| false_06 | exercise | push-ups, workout, sport (false alarm) |

## Counts

Missed: small 4, camera 3, fast 2, other 2, offframe 1
False alarms: lying 2, fast 1, label 1, notperson 1, exercise 1
