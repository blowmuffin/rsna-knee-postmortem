# Submission is slower again?
611482 | Tom | 2025-10-11T20:58 | votes 33

During the last 24 hours, our submissions took less time than before on the sample test set, but all of them timed out on the hidden test set. Has the same issue occurred again?

## Comments
- [Optimo | 2025-10-13T07:38 | v9] Made a sub yesterday with 3 folds which timed out, same with 5 folds ran ok before… @ryanholbrook 
  
  Is there something that can be done to prevent unfair random deadline ?
  - [Optimo | 2025-10-13T12:33 | v3] @evancalabrese  ?
    - [Optimo | 2025-10-13T15:20 | v4] @ryanholbrook  can we hope for increased capacities on the last day to finalize our submissions ?
    - [Tom | 2025-10-13T22:42 | v1] @operedoud2  that's so sad man.
- [Ryan Holbrook | 2025-10-14T19:05 | v9] Hi everyone,
  
  We're investigating the reports of slowdowns in submission times. There are (potentially) two separate issues at play here.
  
  The first is a delay in submission start times due to GPU queueing. During times of high demand, we can run out of GPU capacity. Submissions will then enter a queue until the next GPU is available. This delay doesn't count against the overall submission time limit and wouldn't itself be a cause of submission failures.
  
  The second possible problem is submissions themselves running more slowly than they have in the past. We haven't been able to determine whether this is actually happening in a systematic way due to a fault in our evaluation backend (because of a problem with the machines running the submissions, or a database issue, say), or whether, when it happens, it's due instead to some kind of variance introduced by stochastic submission code. It's not clear that, at least on average, submissions are running more slowly, so the problem is difficult to diagnose.
  
  We will continue to investigate the issue, and if it proves that there is a fault in Kaggle's evaluation system, we will ensure that the affected submissions are rerun.
  - [Optimo | 2025-10-14T19:16 | v5] Thank you for taking a look at this!
    
    You’ll find an example in my latest subs of a 100% deterministic code which timed out 2 days ago and finished within time limit yesterday.
    
    I think there are plenty of competitors with similar examples!
    - [MD The Dramatist | 2025-10-15T01:15 | v0] Thank you!  M
  - [Robin Smits | 2025-10-14T19:18 | v4] @ryanholbrook  There are various mentions from kagglers where the same (or more or less the same…) submission suddenly takes a lot longer or just times out. It looks like the second issue but understandable that can be difficult to analyze.
    
    If required I can provide with some submission with more models finished, other with less models timed out. Likely other Kagglers can also provide you with details if required.
    
    Good luck anyway.
  - [Tom | 2025-10-14T19:29 | v0] All the best.
  - [Cyrus | 2025-10-14T19:42 | v1] Please extend the deadline just a few days
  - [Optimo | 2025-10-14T20:36 | v0] What about submissions made more than 12h before deadline which are still queuing after deadline?
  - [None | 2025-10-14T23:23 | v0] 
  - [None | 2025-10-14T23:24 | v0] 
  - [Tom | 2025-10-14T23:32 | v1] @ryanholbrook  this is our last few submissions.
    
    [img]
    - [None | 2025-10-15T08:46 | v0] 
  - [Stefan Denner | 2025-10-15T08:46 | v2] I might be naive but the situation is quite unfair, imo. Now that also the private LB is open and all submissions also revealed the private dataset score, I don't really see a way of how to bring it back to a fair state again, even if the affected submissions would be rerun… Such a pity that such a nice competition ends like this…
    - [Tom | 2025-10-15T08:56 | v3] @st3v3d  Agreed. Everyone can now see the private score, which means you can no longer allow participants to make another selection or automatically choose the best private score. Also, I’m really curious to see how well our full 5-fold approach would perform, since our submitted winning solution used only 2 folds due to the server slowdown.
    - [YYama | 2025-10-15T14:49 | v4] I completely agree.
      
      Now that the private leaderboard scores have been revealed together with the public ones, it’s no longer possible to make fair re-selections of submissions — everyone already knows the correlation between the two. Even in the unrealistic case that the organizers collected a new external test set, it would still be impossible to restore true fairness at this point.
      
      For future competitions, I think it’s essential to explicitly warn participants to finalize their submissions early, since from my experience, heavy server load and timeouts near the deadline happen quite frequently. The rule should clearly state that deadlines will not be extended and that it’s each participant’s responsibility to complete submissions in advance.
      
      I’d really appreciate it if @ryanholbrook  could kindly consider this for future competitions.
- [Sergio Alvarez | 2025-10-13T01:21 | v5] @ryanholbrook , any confirmation that the servers are running slower? Will this continue until the end of the competition? We tried to verify by submitting an old notebook that previously finished in ~10 hours, but now it’s timing out.
  - [Ángel Jacinto Sánchez Ruiz | 2025-10-13T01:23 | v5] I've got 3 timeouts on previously finished submissions, just weights changed… Sad, I've discovered a bug that could explain my huge CV to test score drop. Reducing ensemble charge to minium at the moment. I hope I can at least check my bug fix.
    
    EDIT: Would be nice if confirmed then timeouts would be rerun just for fairness.
- [Robin Smits | 2025-10-14T17:36 | v6] T.b.h.
  
  I'am not sure what is worse…the submission issues about which kagglers complain already for at least 2 days…
  
  Or the seemingly total lack of response from the Kaggle hosts ….
  
  @inversion  @ryanholbrook  @sohier  anybody?
- [AndresMartinezMora | 2025-10-13T17:04 | v3] Are there any updates regarding this?
- [Stefan Denner | 2025-10-13T06:29 | v3] Hi, we are facing the same issue. Submissions that previously took 8–9 hours are now running into timeouts. This is very unfortunate.
  - [Tom | 2025-10-13T07:18 | v2] @st3v3d  It seems this competition ended two days early.
    - [Stefan Denner | 2025-10-13T10:39 | v2] That would be such a pity🥲
- [Robin Smits | 2025-10-13T15:52 | v4] Same here…very slow…
- [Abhishek Jelloji | 2025-10-13T02:55 | v4] @ryanholbrook  anything on this? Will submissions be rerun or any timeline changes @evancalabrese . Thank you for your info.
- [Eddie | 2025-10-12T09:14 | v4] Seems so. 2 days, submissions took around 2-3 hours. Made 2 submissions last night, one's been running for ~ 7 hours.
  - [Ángel Jacinto Sánchez Ruiz | 2025-10-12T12:10 | v1] Is known that submission time is ofuscated. It will show you randomly larger execution times.
- [Tom | 2025-10-15T01:00 | v1] I want to share that in the last minute before deadline, one 21hr sub finished (best result), and I select it. One of our submission is still running now.
  
  [img]
- [Khushi Yadav | 2025-10-13T14:24 | v1] Yeah, we are facing the same issue.
- [MD The Dramatist | 2025-10-14T06:59 | v-6] Don't you think it's because we are ALL trying to do it at the same time, last minute panic and all that jazz… ?
  - [AndresMartinezMora | 2025-10-14T08:31 | v2] Yes, of course, we are just trying to see if there is any feasible solution to this
- [AC | 2025-10-15T15:29 | v0] Hi Kaggle team,
  
  I submitted my notebooks before the deadline, but they kept running and I only received the scores after the deadline. These were my best submissions, and I couldn’t select them in time. I believe the same issue affected many participants. Had we received the scores before, we'd have selected those. Given how rankings are clustered around 0.69, it’s likely others also made improvements but couldn’t select their final submissions due to issues beyond their control. Would it be possible to consider such submissions internally so no one is unfairly penalized for server delays?
- [AC | 2025-10-14T14:42 | v0] Oh man, it's been running for last 11 hours. I'd to run some final experiments, looks like all came down to 0. Shouldn't have waited for the last day.  🙇‍♀️
- [Sheng-Kai Ku | 2025-10-14T10:50 | v0] What if the deadline has passed but my submission is still scoring? Will it still count? It's a shame to say that our submission wasn't successful at any time, and our Final version is still in training.😭
  - [AndresMartinezMora | 2025-10-14T10:57 | v0] As far as I know the deadline is sharp, since the private leaderboard is published just straightaway after the deadline. If you want to do some final submission that takes almost the limit of 12h, I would recommend to do it already now. It is quite unfortunate…
    - [Sheng-Kai Ku | 2025-10-14T11:21 | v0] OK, thanks for your response. I wil try hard🤧
  - [Mao | 2025-10-14T11:46 | v2] I tried that once, your late submission would not count for public leaderboard. Because you can select two notebook to get scored and you can’t select the ones scoring. Plus, if you let the system choose the best scoring one, the choices are finalized before final leaderboard show up.
    - [Sheng-Kai Ku | 2025-10-14T16:54 | v0] Thanks for your remind. But we just knew that our submit was failed. We got a Kaggle Error again and still don't know why, don't know how to fix. I think it's over. I'm glad to join this competition. Unlucky but learned a lot.🥲
- [Ángel Jacinto Sánchez Ruiz | 2025-10-13T12:43 | v0] I don't remember where but I'm pretty sure I've read execution time would be ofuscated by show randomly larger executions times. I assume that mechanic never would produce time out by itself…
- [Tom | 2025-10-12T12:21 | v0] I just don't know, why same submission finished within 11hr yesterday but needs to run nearly 12hr today.
  - [Ian Pan | 2025-10-12T21:50 | v4] Towards the end of competitions, sometimes the submissions run slower because more people are submitting and this probably slows down the servers.
    - [Tom | 2025-10-12T22:22 | v0] @vaillantali  queue requires a day