"""Top public notebooks by public score (Kaggle web endpoint)."""
import time, requests
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
for sort in ("SCORE_DESCENDING", "DATE_RUN"):
    body = {"kernelFilterCriteria": {"search": "", "listRequest": {"competitionId": comp["id"], "sortBy": sort, "pageSize": 40, "group": "EVERYONE", "page": 1}},
            "detailFilterCriteria": {"deletedAccessBehavior": "RETURN_NOTHING", "unauthorizedAccessBehavior": "RETURN_NOTHING", "excludeResultsFilesOutputs": True}, "readMask": "pinnedKernels"}
    ks = api("kernels.KernelsService", "ListKernels", body).get("kernels", [])
    print('==', sort)
    for k in ks[:40]:
        print(k.get("bestPublicScore"), k.get("scriptVersionDateCreated", k.get("lastRunTime", ""))[:16], k.get("totalVotes"), k["scriptUrl"])
