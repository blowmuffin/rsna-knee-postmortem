"""Public notebooks of the competition with their best public score (Kaggle web endpoint, public data) -> code/scores.csv"""
import time, requests, pandas as pd, json
s = requests.Session(); s.get("https://www.kaggle.com/competitions", timeout=60)
h = {"x-xsrf-token": s.cookies.get("XSRF-TOKEN", "")}
def api(service, method, body):
    for a in range(5):
        r = s.post(f"https://www.kaggle.com/api/i/{service}/{method}", json=body, headers=h, timeout=90)
        if r.status_code == 200: return r.json()
        time.sleep(3 * (a + 1))
    raise RuntimeError(r.status_code)
comp = api("competitions.CompetitionService", "GetCompetition", {"competitionName": "rsna-knee-abnormality-detection"})
comp = comp.get("competition", comp)
raw = {}
for sort in ("SCORE_DESCENDING", "DATE_RUN", "VOTE_COUNT", "DATE_CREATED", "HOTNESS", "COMMENT_COUNT"):
    for page in range(1, 60):
        body = {"kernelFilterCriteria": {"search": "", "listRequest": {"competitionId": comp["id"], "sortBy": sort,
                "pageSize": 100, "group": "EVERYONE", "page": page}},
                "detailFilterCriteria": {"deletedAccessBehavior": "RETURN_NOTHING", "unauthorizedAccessBehavior": "RETURN_NOTHING",
                                         "excludeResultsFilesOutputs": True}, "readMask": "pinnedKernels"}
        got = api("kernels.KernelsService", "ListKernels", body).get("kernels", [])
        if not got: break
        for k in got: raw[k["scriptUrl"]] = k
    print(sort, len(raw))
json.dump(list(raw.values()), open('code/kernels_web.json', 'w'))
print(sorted(next(iter(raw.values())).keys()))
