"""뷰어 HTML + .splat → 파일 하나짜리 HTML.

왜 하나로 합치나 — 남한테 보내려면 서버를 못 띄운다. 브라우저는 file:// 에서
fetch 를 막으므로(CORS) .splat 을 옆에 두는 방식은 상대방 쪽에서 안 열린다.
base64 로 HTML 안에 넣으면 더블클릭만으로 열린다. 용량은 4/3 배가 된다.
"""
import base64, io, os, sys


def build(html_in, splat_in, out_path):
    html = io.open(html_in, encoding="utf-8").read()
    raw = open(splat_in, "rb").read()
    b64 = base64.b64encode(raw).decode("ascii")

    import re
    m = re.search(r'fetch\("([^"]+)"\)', html)
    if not m:
        raise RuntimeError("HTML 에서 fetch 호출을 못 찾았습니다")
    old, asset = m.group(0), m.group(1)

    # <script> 안에 넣으면 브라우저가 실행하려 들지 않는다(type 이 js 가 아님).
    payload = ('<script id="splatdata" type="application/octet-stream">'
               + b64 + '</script>\n')
    loader = """(async () => {
  const s = document.getElementById("splatdata").textContent.trim();
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8.buffer;
})()"""

    html = html.replace(old, loader)

    # fetch 를 없앴으면 그 뒤에 붙은 응답 검사(.then(r => ... r.arrayBuffer()))도
    # 같이 빼야 한다. 남겨두면 ArrayBuffer 에 대고 r.ok 를 물어보게 되고,
    # undefined 라서 "못 읽었습니다" 를 던지며 화면이 빈 채로 끝난다.
    # 정규식은 따옴표·괄호 이스케이프에서 조용히 빗나가므로 위치로 자른다.
    head = ".then(r => { if(!r.ok)"
    tail = "r.arrayBuffer(); })"
    i = html.find(head)
    if i < 0:
        raise RuntimeError("응답 검사 구문을 못 찾았습니다 — 뷰어 HTML 이 바뀌었나?")
    j = html.find(tail, i)
    if j < 0:
        raise RuntimeError("응답 검사 구문의 끝을 못 찾았습니다")
    html = html[:i] + html[j + len(tail):].lstrip()
    if ".then(r =>" in html:
        raise RuntimeError("응답 검사 구문이 남아 있습니다")

    html = html.replace("<script>\n// 3D 가우시안", payload + "<script>\n// 3D 가우시안")

    io.open(out_path, "w", encoding="utf-8", newline="\n").write(html)
    print(f"{out_path}  ({os.path.getsize(out_path)/1e6:.1f}MB)")
    return out_path


if __name__ == "__main__":
    build(*sys.argv[1:4])
