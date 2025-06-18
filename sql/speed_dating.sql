SELECT
    -- Self
    iid
    , age
    , gender
    , mn_sat
    , income
    , goal
    , date_frequency
    , go_out
    , exphappy -- how happy do you expect to be with the people, taken before the event
    , expnum -- how many people do you think will be interested in you, taken before the event
    , match_es -- how many matches do you estimate, taken at the end of the night
    , satis_2 -- how satisfied were you with the people you met, taken the day after

    -- Attribute views
    , attr1_1 -- what you look for in partner
    , amb1_1
    , fun1_1
    , intel1_1
    , shar1_1
    , sinc1_1
    , attr4_1 -- what you think your gender looks for
    , amb4_1
    , fun4_1
    , intel4_1
    , shar4_1
    , sinc4_1
    , attr2_1 -- what you think opposite gender looks for
    , amb2_1
    , fun2_1
    , intel2_1
    , shar2_1
    , sinc2_1
    , attr3_1 -- how do you rate yourself
    , amb3_1
    , fun3_1
    , intel3_1
    , sinc3_1
    , attr5_1 -- how do you think others rate you?
    , amb5_1
    , fun5_1
    , intel5_1
    , sinc5_1

    -- Partner Scorecard
    , decision -- views on partner
    , attr
    , amb
    , fun
    , intel
    , shar
    , sinc
    , like_partner -- how much do you like this person
    , prob -- how probable do you think it is that this person will say yes

    -- Partner
    , pid

    -- Match
    , match_flag -- did both say yes
    , int_corr -- correlation of interests
    , samerace -- same race

FROM speed_dating
ORDER BY iid, pid;