# Server load has a substantial influence on code runtime.
611721 | xun1993 | 2025-10-14T01:41 | votes 11

My code that used to run in 8 hours has now been running for over 10 hours without results. It's probably because the same machine is running several people's code simultaneously. But is this 12-hour time limit really reasonable? It's easily affected by the machine's load, which isn't something we can control.

## Comments
- [xun1993 | 2025-10-14T02:16 | v6] @ryanholbrook  It's extremely unfair that our final, fully-optimized version timed out due to server issues
  - [Mao | 2025-10-14T11:42 | v0] I think the situation might worse in the few last day of the comeptition
    - [Taha_Alshatiri | 2025-10-14T11:48 | v0] what are you talking about? 11 hours left
    - [Mao | 2025-10-14T12:27 | v0] Yeah, so is even worse…… i don’t even think the public leaderboard is going to change.
- [xun1993 | 2025-10-14T13:17 | v1] I've run 7 submissions over the past two days without a single result. The longest one took 18 hours. I set up my submission to stop processing and return default values if it exceeds 11 hours and 55 minutes. Even with that precaution, some runs still went over 14 hours. The experience has been terrible, and the official platform hasn't given any explanation. @ryanholbrook
- [Ángel Jacinto Sánchez Ruiz | 2025-10-14T09:39 | v2] That is hilarius is a possible variance between 8 hours and timeout.
  - [None | 2025-10-14T13:14 | v0] 
  - [Navneet | 2025-10-15T01:42 | v-1] 8 hours and a timeout!
- [Reidner Santos | 2025-10-14T13:49 | v0] I have two submissions that have been running for 12 hours, but they should have finished in about 9 hours.