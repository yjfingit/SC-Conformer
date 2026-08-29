"""Baidu-share direct downloader (bypasses PCS API restriction).

Flow: shorturlinfo -> verify(pwd) -> share/list (walk dirs) ->
share/download (dlink) -> ranged GET. Uses BDUSS+STOKEN cookies.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import concurrent.futures as cf

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
SHARE_URL = "https://pan.baidu.com/s/1-w8AdRRnbCxx13548VvwRQ"
PWD = "1390"
BDUSS = ("c4NzA2NzhQS0s3fnZPV2JvNWU3UERmd2l1NENJbFlBaDM1dH5Ka1FYMU5jS0px"
         "SVFBQUFBJCQAAAAAAQAAAAEAAADsb9uWbGVsZWxleXlf9sEAAAAAAAAAAAAAAAAA"
         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE3jempN43pq"
         "aW")
STOKEN = "f5ab4fa014fe76ce3c080880d4507d9d53551ea5a21a10e9c9b99946b436d326"
OUT = "/root/autodl-tmp/ICLR/data/baidu_share"

env_no_proxy = {k: v for k, v in os.environ.items()
                if "proxy" not in k.lower()}


def http(url, data=None, headers=None, method=None, timeout=30):
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    h["Cookie"] = f"BDUSS={BDUSS}; STOKEN={STOKEN}"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout).read()


def jget(url, **kw):
    return json.loads(http(url, **kw).decode())


def main():
    os.makedirs(OUT, exist_ok=True)
    surl = SHARE_URL.split("/s/1")[1]
    print("surl:", surl)

    info = jget(f"https://pan.baidu.com/api/shorturlinfo?app_id=250528"
                f"&shorturl={surl}&root=1")
    if info.get("errno") != 0:
        # some shares need verify first
        print("shorturlinfo errno:", info.get("errno"))
    uk = info.get("uk") or info.get("share_uk")
    shareid = info.get("shareid") or info.get("share_id")
    sign, timestamp = info.get("sign"), info.get("timestamp")
    print("uk:", uk, "shareid:", shareid, "sign?", bool(sign))
    if sign is None:
        raise SystemExit("shorturlinfo did not provide sign; errno="
                         + str(info.get("errno")))

    # verify password -> randsk cookie
    ver = jget(f"https://pan.baidu.com/share/verify?surl={surl}&bdstoken=null"
               f"&t={int(time.time()*1000)}&channel=chunlei&web=1"
               f"&app_id=250528&clienttype=0",
               data=urllib.parse.urlencode({"pwd": PWD}).encode())
    print("verify errno:", ver.get("errno"))
    randsk = ver.get("randsk", "")
    cookie = f"BDUSS={BDUSS}; STOKEN={STOKEN}; BDCLND={urllib.parse.quote(randsk)}"

    files = []
    seen_dirs = set()

    def list_dir(path):
        q = urllib.parse.urlencode({
            "shareid": shareid, "uk": uk, "root": 1 if path == "" else 0,
            "dir": path, "randsk": randsk, "web": 1, "app_id": 250528,
            "clienttype": 0, "page": 1, "num": 1000,
            "order": "name", "desc": 0, "showempty": 0})
        r = jget(f"https://pan.baidu.com/share/list?{q}")
        if r.get("errno") != 0:
            print("list errno", r.get("errno"), "at", repr(path))
            return []
        return r.get("list", [])

    stack = [""]
    while stack:
        d = stack.pop()
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        for f in list_dir(d):
            if f.get("isdir") == "1" or f.get("isdir") == 1:
                stack.append(f["path"])
            else:
                files.append(f)
        time.sleep(0.2)
        if len(files) % 50 == 0 and files:
            print("found", len(files), "files...", flush=True)
    print("total files:", len(files))
    tot = sum(f.get("size", 0) for f in files)
    print(f"total size: {tot/1e9:.2f} GB")
    with open(os.path.join(OUT, "_manifest.json"), "w") as fo:
        json.dump(files, fo, ensure_ascii=False, indent=1)
    print("manifest ->", os.path.join(OUT, "_manifest.json"))


if __name__ == "__main__":
    main()
