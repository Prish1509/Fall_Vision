# Complete system on whole videos (Steps 23 and 25)

detect_falls.py scored end to end on the cached skeletons: every tracked person is scored, a video alarms if any person does. Threshold 0.849 (chosen on validation).

| Set | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Fall-start error | Alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_gmdcsa24 | 80 | 0.916 | 0.844 | 1.000 | 0.912 | 38 | 7 | 35 | 0 | 0.4 s | 1.4 s |
| test_staged | 57 | 0.968 | 0.968 | 0.968 | 0.965 | 30 | 1 | 25 | 1 | 0.2 s | 1.4 s |
| test_oops | 818 | 0.875 | 0.945 | 0.815 | 0.790 | 604 | 35 | 42 | 137 | 0.5 s | 1.0 s |

Mistakes: 181

| Video | True | Predicted | Score |
|---|---|---|---|
| caucafall/adl/PickupobjectS8 | no fall | alarm | 0.865 |
| le2i/Home_01/video_18 | fall | no alarm | 0.000 |
| GMDCSA24/Subject_3/ADL/04 | no fall | alarm | 0.948 |
| GMDCSA24/Subject_3/ADL/07 | no fall | alarm | 0.852 |
| GMDCSA24/Subject_3/ADL/08 | no fall | alarm | 0.986 |
| GMDCSA24/Subject_4/ADL/07 | no fall | alarm | 0.986 |
| GMDCSA24/Subject_4/ADL/10 | no fall | alarm | 0.965 |
| GMDCSA24/Subject_4/ADL/15 | no fall | alarm | 0.982 |
| GMDCSA24/Subject_4/ADL/16 | no fall | alarm | 0.973 |
| OOPS/falls/29BestWorkFailNominees-FailArmyHallofFameAugust201722 | no fall | alarm | 0.977 |
| OOPS/falls/29HilariousInstantKarmaFailNominees-FailArmyHallofFame26 | fall | no alarm | 0.000 |
| OOPS/falls/31BackToSchoolFailNominees-FailArmyHallofFameSeptember201743 | fall | no alarm | 0.000 |
| OOPS/falls/34FunnyKidNominees-FailArmyHallOfFameMay20179 | no fall | alarm | 0.990 |
| OOPS/falls/34WorkoutFailNominees-FailArmyHallOfFameApril201730 | no fall | alarm | 0.995 |
| OOPS/falls/38ParkourFailNominees-FailArmyHallOfFameMay20176 | fall | no alarm | 0.000 |
| OOPS/falls/ABoatloadofFails-ThrowbackFailsNovember2017_FailArmy18 | fall | no alarm | 0.000 |
| OOPS/falls/AChristmasMiracle-FailsYouMissed21December2017_FailArmy3 | no fall | alarm | 0.946 |
| OOPS/falls/AChristmasMiracle-FailsYouMissed21December2017_FailArmy80 | fall | no alarm | 0.000 |
| OOPS/falls/AlleyOOPS-FailsoftheWeekOctober201710 | fall | no alarm | 0.000 |
| OOPS/falls/AngerManagement-FailsoftheWeekJanuary2019_FailArmy2 | fall | no alarm | 0.000 |
| OOPS/falls/AntsAreMarching-FailsoftheMonthOctober2017_FailArmy7 | fall | no alarm | 0.000 |
| OOPS/falls/AreYouSerious-ThrowbackThursdaySeptember2017_FailArmy75 | fall | no alarm | 0.000 |
| OOPS/falls/ArmpitWax-FailsYouMissedApril2018_201880 | fall | no alarm | 0.000 |
| OOPS/falls/BackintheMud-DownandDirtyFailsOctober2018_FailArmy9 | fall | no alarm | 0.000 |
| OOPS/falls/BadHairDay-FailsoftheweekApril2019_FailArmy11 | fall | no alarm | 0.000 |
| OOPS/falls/BasketballFails-MarchMadnessMarch2017_FailArmy32 | no fall | alarm | 0.972 |
| OOPS/falls/BatsBallsFailCompilation_ByFailArmy201622 | no fall | alarm | 0.947 |
| OOPS/falls/BatsBallsFailCompilation_ByFailArmy201631 | fall | no alarm | 0.000 |
| OOPS/falls/BatsBallsFailCompilation_ByFailArmy201650 | no fall | alarm | 0.866 |
| OOPS/falls/BeeKeeperBusiness-FailsoftheWeekNovember2018_FailArmy2 | fall | no alarm | 0.000 |
| OOPS/falls/BeeKeeperBusiness-FailsoftheWeekNovember2018_FailArmy9 | no fall | alarm | 0.979 |
| OOPS/falls/BestFailoftheWeek-BadKittyApril2018_FailArmy13 | no fall | alarm | 0.985 |
| OOPS/falls/BestFailsOfTheMonth-ICantHearFebruary2018_FailArmy15 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek1July2016_FailArmy8 | no fall | alarm | 0.954 |
| OOPS/falls/BestFailsofWeek2June2016_FailArmy34 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek4June2016_FailArmy25 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek4June2016_FailArmy60 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsoftheMonth-DrivingMeCrazyJanuary2018_FailArmy41 | no fall | alarm | 0.998 |
| OOPS/falls/BestFailsoftheMonthApril2016_FailArmy20 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsoftheMonthFebruary2016_Babe_YoureActuallyCrazyFailArmy0 | no fall | alarm | 0.858 |
