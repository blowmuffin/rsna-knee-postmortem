"""Public notebooks of the competition with best public score etc. (Kaggle web endpoint, public data).
python -I fetch_kernels.py <out.json>   (read-only POSTs to the listing endpoint)"""
import sys, time, json, requests
s = requests.Session(); s.get("https://www.kaggle.com/competitions", timeout=60)
h = {"x-xsrf-token": s.cookies.get("XSRF-TOKEN", "")}


def api(service, method, body):
    for a in range(6):
        r = s.post(f"https://www.kaggle.com/api/i/{service}/{method}", json=body, headers=h, timeout=90)
        if r.status_code == 200: return r.json()
        time.sleep(4 * (a + 1))
    raise RuntimeError(r.status_code)


comp = api("competitions.CompetitionService", "GetCompetition", {"competitionName": "rsna-knee-abnormality-detection"})
comp = comp.get("competition", comp)
raw = {}
for sort in ("SCORE_DESCENDING", "DATE_RUN", "DATE_CREATED", "VOTE_COUNT", "HOTNESS", "COMMENT_COUNT"):
    for page in range(1, 60):
        body = {"kernelFilterCriteria": {"search": "", "listRequest": {"competitionId": comp["id"], "sortBy": sort,
                "pageSize": 100, "group": "EVERYONE", "page": page}},
                "detailFilterCriteria": {"deletedAccessBehavior": "RETURN_NOTHING", "unauthorizedAccessBehavior": "RETURN_NOTHING",
                                         "excludeResultsFilesOutputs": True}, "readMask": "pinnedKernels"}
        got = api("kernels.KernelsService", "ListKernels", body).get("kernels", [])
        if not got: break
        for k in got: raw[k["scriptUrl"]] = k
        time.sleep(0.7)
    print(sort, len(raw), flush=True)
json.dump(list(raw.values()), open(sys.argv[1], 'w'))
print(sorted(next(iter(raw.values())).keys()))
