# What not to do: lessons from a competition we lost

**From:** the senior who ran our last Kaggle competition.
**To:** whoever runs the next one, human or agent. Read it before you start, and re-read it at every weekly review.

**Our result:**
- Public 0.958, private 0.942. Rank 1,774 of 6,425, where gold needed 0.954.
- The winners scored 0.960, and most of them were small teams.
- We were not short on effort. We had a clean pipeline, tests, and a log of every experiment.

**We lost because we were very precise about the wrong question.**
- We measured our work against a flawed stand-in for the real target.
- We dismissed the one idea that mattered without trying it.
- We spent the final week copying public notebooks.

Each rule below comes from a mistake we actually made.

---

## The 12 rules

### 1. Don't start by copying the leaderboard. Start by finding the gap.
On day 1, write down three things:
- (a) how the **test target** is produced;
- (b) how your **training target** is produced;
- (c) every way the two differ: who labelled, from what evidence, by which definitions, and at what threshold.

The biggest difference is usually the biggest lever. Then ask: "If my training labels were perfect, how much would I gain?" If the answer is "a lot", the competition is about data and labels, not models.

*Our case:* the training labels were mined from text reports, but the test was graded by experts looking at the images. Every winner fixed that gap. We tuned models on top of it.

### 2. Don't call a plateau "noise".
When five variants of one approach all land "within noise" of each other, the approach is saturated. The noise is showing you its ceiling. Change what goes into the approach (data, labels, resolution, inputs), not its knobs.

*Our case:* eight label sets, all built from the same text, scored 0.887–0.910 on the expert subset. We logged them as "all within noise". Labels built from a different source of evidence scored 0.935–0.956.

### 3. Don't reject an idea based on how you imagine building it.
Any idea with a credible signal (a forum post, a small experiment, a paper on the same data) gets a **2-hour spike** with the cheapest possible implementation before it can be rejected. Log the spike's result, not your guess.

*Our case:* we wrote off "have an LLM read the training images and label them" as "not feasible in 6 days", because we pictured building it inside the competition platform. The winners ran it from a laptop in a few hours, using the same AI command-line tool we were working in.

### 4. Don't validate against a measure that shares your training data's bias.
- If your validation target comes from the same source as your training labels, it can never show you that the labels are the limit.
- Build a validation set whose target is as close as possible to how the test is graded.
- Then check, over 10 or more submissions, that your local score ranks submissions the same way the leaderboard does.
- If it doesn't, fix the validation before you tune anything else. Group folds by the source of hidden correlation (site, user, time, device).

*Our case:* our main CV scored agreement with text-mined labels. It rewarded models for copying the reports' blind spots.

### 5. Use a small expert-labelled set carefully: don't rely on it alone, and don't ignore it.
- Tune on half of it and check on the other half.
- Score candidate label sets by AUC **and** log loss **and** the count of confident errors. Two label sets with the same AUC can differ greatly in quality.
- Know its noise: on 58 items it was about ±0.02.

### 6. Don't run experiments that can't detect the effect you care about.
- Before any test, write down the effect size you expect.
- Then check that the test can actually see an effect that size. Displayed precision, split size and blend weight all limit what you can see.
- If it can't, redesign the test: a larger weight, a paired comparison, or several runs.

*Our case:* we added our own models to a public blend at 5% weight, twice, and read a change in the third decimal place both times. We concluded "our models hurt". At 5% weight, the expected effect was smaller than the precision the leaderboard displays.

### 7. Don't make a copied public notebook your final submission.
- Exact copies tie, ties go to whoever submitted first, and a whole block of copies rises or falls together on the private board.
  - *Our case:* 1,322 teams had the identical private score, and only the first ~27 of them got a medal.
- Public notebooks are tuned against the public leaderboard and drop more on the private one.
  - *Our case:* −0.016 for the public notebooks, against about −0.010 for the winners' own models.
- Use public notebooks to read and as ingredients, never as your picks.
- A notebook's score badge is the **best score of any version it ever had**, not the score of its current code. Check the version history before you trust it. That one cost us a day and two submissions.

### 8. Don't spend the last week on blending.
- Ensembles added at most 0.002 for the winners; better labels added 0.006–0.015.
- If your end-game plan is "weights, unions, stacking", you have already given up on the lever that decides the result.
- Hand-set simple weights. Weights tuned on a noisy leaderboard lost the winners 0.002–0.007 on the private board.

### 9. Don't let free compute decide your model size.
- Check what resolution, model size and number of epochs the strong public single models use, and match it.
- Price a rented GPU on day 1: ~100 GPU-hours (~$100) trained a winning ensemble.
- Train wherever it's cheap, and use the competition platform only for inference.

*Our case:* we trained at 224–288 px for 6 epochs because the free quota allowed about 4 hours per fold. The winners trained at 384 px for 20–25 epochs.

### 10. Don't go dark on the leaderboard.
- Submit your **own** pipeline at least every two days. A submission is the cheapest honest measurement you have.
- Use the daily submission allowance as probes, each with a written hypothesis.

*Our case:* we went 11 days without submitting our own pipeline, and finished with 18 submissions to the winners' 100–240.

### 11. If you join late, don't plan as if you joined early.
With three weeks left:
- spend 2 days finding the single biggest lever;
- spend 80% of the remaining time on it;
- skip broad literature reviews and wide parameter sweeps.

*Our case:* we joined with about 3 weeks left and still did research reports and wide sweeps. A literature review had concluded "no known technique gives a big step". It was wrong: the step was in the competition's own data.

### 12. Don't let the run depend on one session staying alive.
Always have:
- **(a)** your best *unique* submission already selected as a final pick, replaced as better ones land;
- **(b)** a written next-step list that someone else can carry out cold;
- **(c)** internal deadlines one day earlier than the real ones.

*Our case:* the working session ended four days before the deadline. Two finished submissions were never sent, no picks were chosen, and the platform picked our copies for us.

---

## Smaller lessons
- **Use AI agents as labour.** Headless agent runs (`claude -p --json-schema`, `codex exec`, one isolated workspace per item) can label, draw boxes and audit data at thousands of items a day on subscription quota. Check the rules first (here the host explicitly allowed it), then do it in week 1.
- **Read the rules on external data, pretrained weights and LLM use in week 1.** Ask the host in the forum at once. Don't spend the final week on compliance worries, and don't build on components you might have to drop.
- **When someone reports a large gain on the holdout or expert set, reproduce it within a day.**
- **Look at the gap between the top of the leaderboard and the public notebooks.** Whatever the top teams have that the public notebooks don't is the lever. Papers rarely tell you.
- **Hidden-test runtime varies from run to run.** Keep at least 20% margin under the time limit, and give each stage a time budget with a fallback.
- **Know the public split's noise** (ours: SD ≈ 0.003). Smaller public differences are coin flips, so don't chase third-decimal gains.
- **Keep doing what we did right:** log every result the moment it lands, check every run before submitting it, write the hypothesis in each submission message, and keep a handoff file a fresh session can resume from.

---

## Week-by-week template
| When | Focus | Done when |
|---|---|---|
| Week 1 | Rules, data audit, **gap analysis** (rule 1), compute budget, first submission of your own pipeline, validation design (rule 4) | Gap note written; own pipeline on the leaderboard; rented GPU priced |
| Week 2 | Spike the biggest lever (usually data or labels), then scale it | Lever measured on the leaderboard with your own pipeline |
| Middle weeks | Inputs and scale: resolution, crops, epochs, model size. Own submission every 2 days | Validation tracks the leaderboard; best single model improving |
| Last 2 weeks | Diversity from **different inputs or labels**, not just backbones; simple ensemble; picks kept current | Two unique picks selected |
| Last 4 days | Freeze. No new pipelines. Runtime margin checked. Final picks: one strongest unique submission + one diverse hedge | Picks selected 48 h before the deadline |

## Red flags: stop and rethink when you see one
| You notice | It means |
|---|---|
| "All variants are within noise" | The approach is saturated; change its input (rule 2) |
| "Not feasible in N days", with no spike behind it | Untested assumption (rule 3) |
| Local score and leaderboard disagree for 3+ submissions | Wrong validation target (rule 4) |
| A component tested at < 10% weight | The test can't see the effect (rule 6) |
| Your best submission is a public notebook | You are in a tie block (rule 7) |
| No submission of your own pipeline in 3 days | Flying blind (rule 10) |
| The final-week plan says "blend / union / weights" | Wrong lever (rule 8) |
| Final picks not selected 48 h before the deadline | Auto-pick risk (rule 12) |

## For the agent session reading this
- Before each major decision, check it against the 12 rules and the red flags, and name the rule that applies.
- First deliverable: the gap note from rule 1. It should cover:
  - test-target origin vs training-target origin;
  - the rules on external data and LLM use;
  - the compute and submission budget;
  - the one lever you will spike first.
- Keep an experiment log and a handoff file. From week 2 on, the handoff names the current two final picks.
