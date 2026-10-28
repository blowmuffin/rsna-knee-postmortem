# Forum threads read through the Kaggle MCP (summarised; not saved verbatim)

All from rsna-intracranial-aneurysm-detection, read 2026-10-17.

## 611750 "Best submission manual selection" (2025-10-14, 3 comments)
- A participant asked whether the 2 final picks can still be chosen on Oct 15 after seeing results (he feared the "autoselect" would choose for him). One reply (author role not shown in the data, plausibly staff) said selection is open only until the deadline day (Oct 14, 23:59 UTC) and that every pick must be fully scored by then: anything still running at 23:59 does not count. Another reply: pick the 2 you believe will do best on the hidden test, before the reset.

## 608179 "Question about external dataset restrictions" (2025-09-18 .. 09-24, 6 comments)
- Question: do the Large Intracranial Aneurysm Dataset (Zenodo, needs a request form and approval) and ADAM 2020 (request form, confidentiality terms) qualify as external data?
- Host ruling on 2025-09-22 (Evan Calabrese): NO. Large IA requires institution/organisation/PI details and the host's own request without them was rejected; ADAM's agreement limits use to preparing an entry to that challenge.
- Follow-up: the Lausanne TOF-MRA and Royal Brisbane TOF-MRA sets on OpenNeuro (CC0, direct download) were confirmed by the host (2025-09-24) as fully open and usable.
- Takeaway: the host answered with a concrete ruling in 4 days and tested the access path itself (they filed a request). Gate-keeping criteria used: access without institutional affiliation or special permission, and no use restriction.

## 611652 "What's the possibility of a shakeup?" (2025-10-13)
- Poster: moving to a "clinically correct, standardised" pipeline cost 0.04 of validation AUC, taken as a sign that public notebooks fit a particular setting. Reply: public models mostly exploit class/modality imbalance (aneurysm rate about 50% on CTA vs about 25% on MRI T1/T2) rather than finding aneurysms; the same imbalance carries to the test set, so little shake-up expected. (Result: top ordering stable, see shakeup_numbers.md.)

## 603679 "CV vs LB thread" (2025-09-03 .. 10-03)
- Local CV 0.63-0.68 vs LB 0.67-0.70 for 2D EffNetV2 five-fold models; "not stable". A reply recommended stratified K-fold on the Aneurysm Present label; another poster had a data-leak problem with a split that mixed series of one patient.

## 602885 "An Update on Submission Scoring" (2025-08-30, staff)
- Hidden evaluation set updated again; all submissions from 08/20 to 08/30 rescored; the dataset size had stressed their systems. Comment: just loading DICOMs took >4 h for one participant; a submission showing "scoring" for 3 hours.
- Related: 603990 (rescored 08/20-08/27 submissions, done 09-05); 600908 (host decided NOT to rescore after the dataset update because <5% of cases changed).
