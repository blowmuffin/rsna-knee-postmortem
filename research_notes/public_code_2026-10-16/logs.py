"""Fetch only the run log of each notebook in code/log_list.txt via list_kernel_session_output (page 1 carries .log) -> code/logs/<a>__<s>.log"""
import os, time, json
from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
k = KaggleClient(api_token=open(os.path.expanduser('~/.kaggle/access_token')).read().strip()).kernels.kernels_api_client
os.makedirs('code/logs', exist_ok=True)
for ref in open('code/log_list.txt').read().split():
    out = 'code/logs/' + ref.replace('/', '__') + '.log'
    if os.path.exists(out): continue
    user, slug = ref.split('/')
    for i in range(6):
        try:
            r = ApiListKernelSessionOutputRequest(); r.user_name = user; r.kernel_slug = slug; r.page_size = 50
            res = k.list_kernel_session_output(r)
            open(out, 'w', encoding='utf-8').write(res.log or '')
            files = [f.file_name for f in (res.files or [])][:60]
            json.dump(files, open(out + '.files.json', 'w'))
            print(ref, 'ok', len(res.log or ''), 'chars', len(files), 'files', flush=True); break
        except Exception as e:
            print(ref, 'retry', str(e)[:120], flush=True); time.sleep(10 * (i + 1))
