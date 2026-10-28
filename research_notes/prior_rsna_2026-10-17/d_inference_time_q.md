# Compilation of Questions on Inference Time. Official help, please?
607923 | MakeLoveAndPeace | 2025-09-16T22:43 | votes 5

There have been a number of questions on inference time in this forum. However, they were either unanswered or not verified by the official organizers, leaving much confusion on the available inference time still:

- 17 seconds inference per case (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/602640)

- private vs public LB evaluation (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/606980)

- inference time per case (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/603707)

- dicom loading time and 12 hours (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/599253)

- notebook timeout error local vs 12 hours (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/603776)

I think they can be summarized by the following 3 underlying questions:

Q1: on average, how much time should we reserve for one test case for inference? Is it the estimated 17 seconds per case?

Q2: how do we know a successful submission to the public LB will not result in 12-hour timeout error in the final submission?

Q3: will the private and public testsets be evaluated in parallel and each will have 12-hour time limit?

Can the organizers please help clarify on the available inference time? Right now I think many people are worried about the 12 hours error without clear information.

## Comments
- [Ryan Holbrook | 2025-09-18T13:56 | v6] Hi @makelove ,
  
  Q1: From the Data page: "There are nearly 2500 series in the test set." You should divide up the 12 hour allowed runtime across these ~2500 instances. You could consider a "failsafe" mechanism that just submits a default prediction if your submission approaches the 12 hour limit.
  
  Q2: Every submission to this competition runs on the full test set, both public and private. This is in contrast to competitions like Jane Street  which reruns submissions on a new private set after the competition ends. In other words, in this competition, if your submission succeeds on the public LB, it has also already succeeded on the private LB. The private LB scores will be revealed (but not computed) once the competition ends.
  
  Q3: The combined runtime on both test sets is 12 hours. Your submission makes predictions on all test instances in sequence, randomly interleaved between public and private. Your submission doesn't know which are public and which are private.
  - [MakeLoveAndPeace | 2025-09-19T15:02 | v1] Thank you very much for your detailed clarification! It is very helpful!
  - [Suman Kumar Gangopadhyay | 2025-10-02T05:18 | v0] @ryanholbrook  approximately, how many image files can there be in each of these 2500 series
- [Evan Calabrese | 2025-09-17T13:06 | v1] Unfortunately I don't know the answers to these questions. We really need input from @ryanholbrook . I will try to point his attention towards this thread.