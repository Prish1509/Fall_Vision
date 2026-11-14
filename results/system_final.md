# Complete system on whole videos (Steps 23 and 25)

detect_falls.py scored end to end on the cached skeletons: every tracked person is scored, a video alarms if any person does. Threshold 0.824 (chosen on validation).

| Set | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Fall-start error | Alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_gmdcsa24 | 80 | 0.900 | 0.857 | 0.947 | 0.900 | 36 | 6 | 36 | 2 | 0.5 s | 1.7 s |
| test_staged | 57 | 0.968 | 0.968 | 0.968 | 0.965 | 30 | 1 | 25 | 1 | 0.3 s | 1.3 s |
| test_oops | 818 | 0.860 | 0.954 | 0.783 | 0.769 | 580 | 28 | 49 | 161 | 0.4 s | 1.1 s |

Mistakes: 199

| Video | True | Predicted | Score |
|---|---|---|---|
| caucafall/adl/PickupobjectS8 | no fall | alarm | 0.980 |
| le2i/Home_01/video_18 | fall | no alarm | 0.000 |
| GMDCSA24/Subject_3/ADL/08 | no fall | alarm | 0.978 |
| GMDCSA24/Subject_4/ADL/05 | no fall | alarm | 0.928 |
| GMDCSA24/Subject_4/ADL/06 | no fall | alarm | 0.990 |
| GMDCSA24/Subject_4/ADL/07 | no fall | alarm | 0.993 |
| GMDCSA24/Subject_4/ADL/10 | no fall | alarm | 0.944 |
| GMDCSA24/Subject_4/ADL/15 | no fall | alarm | 0.971 |
| GMDCSA24/Subject_4/Fall/09 | fall | no alarm | 0.000 |
| GMDCSA24/Subject_4/Fall/13 | fall | no alarm | 0.000 |
| OOPS/falls/29BestWorkFailNominees-FailArmyHallofFameAugust201722 | no fall | alarm | 0.910 |
| OOPS/falls/29HilariousInstantKarmaFailNominees-FailArmyHallofFame26 | fall | no alarm | 0.000 |
| OOPS/falls/31BackToSchoolFailNominees-FailArmyHallofFameSeptember201743 | fall | no alarm | 0.000 |
| OOPS/falls/31SeniorCitizenFails-FailArmyHallofFameSeptember201728 | fall | no alarm | 0.000 |
| OOPS/falls/34FunnyKidNominees-FailArmyHallOfFameMay20179 | no fall | alarm | 0.988 |
| OOPS/falls/34WorkoutFailNominees-FailArmyHallOfFameApril201730 | no fall | alarm | 0.994 |
| OOPS/falls/38ParkourFailNominees-FailArmyHallOfFameMay20176 | fall | no alarm | 0.000 |
| OOPS/falls/ABoatloadofFails-ThrowbackFailsNovember2017_FailArmy0 | fall | no alarm | 0.000 |
| OOPS/falls/ABoatloadofFails-ThrowbackFailsNovember2017_FailArmy18 | fall | no alarm | 0.000 |
| OOPS/falls/AChristmasMiracle-FailsYouMissed21December2017_FailArmy3 | no fall | alarm | 0.915 |
| OOPS/falls/AChristmasMiracle-FailsYouMissed21December2017_FailArmy80 | fall | no alarm | 0.000 |
| OOPS/falls/AlleyOOPS-FailsoftheWeekOctober201710 | fall | no alarm | 0.000 |
| OOPS/falls/AngerManagement-FailsoftheWeekJanuary2019_FailArmy2 | fall | no alarm | 0.000 |
| OOPS/falls/AntsAreMarching-FailsoftheMonthOctober2017_FailArmy7 | fall | no alarm | 0.000 |
| OOPS/falls/AreYouSerious-ThrowbackThursdaySeptember2017_FailArmy75 | fall | no alarm | 0.000 |
| OOPS/falls/ArmpitWax-FailsYouMissedApril2018_201880 | fall | no alarm | 0.000 |
| OOPS/falls/BackintheMud-DownandDirtyFailsOctober2018_FailArmy9 | fall | no alarm | 0.000 |
| OOPS/falls/BadHairDay-FailsoftheweekApril2019_FailArmy11 | fall | no alarm | 0.000 |
| OOPS/falls/BasketballFails-MarchMadnessMarch2017_FailArmy32 | no fall | alarm | 0.948 |
| OOPS/falls/BatsBallsFailCompilation_ByFailArmy201631 | fall | no alarm | 0.000 |
| OOPS/falls/BatsBallsFailCompilation_ByFailArmy201649 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailoftheWeek-BadKittyApril2018_FailArmy13 | no fall | alarm | 0.995 |
| OOPS/falls/BestFailsOfTheMonth-ICantHearFebruary2018_FailArmy15 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek1July2016_FailArmy39 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek1July2016_FailArmy57 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek2June2016_FailArmy34 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek3June2016_FailArmy57 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsofWeek4June2016_FailArmy25 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsoftheMonth-DrivingMeCrazyJanuary2018_FailArmy3 | fall | no alarm | 0.000 |
| OOPS/falls/BestFailsoftheMonth-DrivingMeCrazyJanuary2018_FailArmy41 | no fall | alarm | 0.997 |
