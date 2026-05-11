import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";

const BODY_CONNS = [
  [0,1],[1,2],[2,81],[81,83],[0,86],[86,95],[95,98],[98,99],
  [86,87],[87,90],[90,91],[2,42],[42,45],[45,48],[48,53],
  [2,3],[3,6],[6,9],[9,14]
];
const BAT_CONNS = [[200,201],[201,202],[202,203],[203,204]];
const ALL_CONNS = [...BODY_CONNS, ...BAT_CONNS];
const BODY_JOINTS = [0,1,2,3,6,9,14,42,45,48,53,81,83,86,87,90,91,95,98,99];
const BAT_JOINTS = [200,201,202,203,204];
const ALL_JOINTS = [...BODY_JOINTS, ...BAT_JOINTS];
const COLORS = [0x00ff88,0xff6644,0x44aaff,0xffcc00,0xff44aa,0x44ffcc,0xaa44ff,0xff8844,0x88ff44,0x4488ff,0xff4444,0x44ff44,0x4444ff,0xffff44,0xff44ff,0x44ffff,0xffffff,0xaaaaaa,0x66ffaa,0xffaa66,0xaa66ff,0x66aaff,0xff66aa,0xaaff66,0xff9999];
const BAT_COLOR = 0xffdd44;

const bodyBoneGeo = (() => { const g = new THREE.CylinderGeometry(0.035, 0.035, 1, 5, 1); g.translate(0,0.5,0); g.rotateX(Math.PI/2); return g; })();
const batBoneGeo = (() => { const g = new THREE.CylinderGeometry(0.06, 0.04, 1, 6, 1); g.translate(0,0.5,0); g.rotateX(Math.PI/2); return g; })();
const jointGeo = new THREE.SphereGeometry(0.06, 6, 4);
const headGeo = new THREE.SphereGeometry(0.14, 8, 6);
const batJointGeo = new THREE.SphereGeometry(0.045, 6, 4);
const batKnobGeo = new THREE.SphereGeometry(0.07, 6, 4);
const batBarrelGeo = new THREE.SphereGeometry(0.055, 6, 4);

export default function App() {
  const sceneRef = useRef(null);
  const [frames, setFrames] = useState(null);
  const [fileConns, setFileConns] = useState(null);
  const [fileJoints, setFileJoints] = useState(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [info, setInfo] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const playRef = useRef(false);
  const speedRef = useRef(1);
  const frameRef = useRef(0);
  const framesRef = useRef(null);
  const connsRef = useRef(ALL_CONNS);
  const jointsRef = useRef(ALL_JOINTS);
  const pmRef = useRef([]);

  useEffect(() => { playRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { frameRef.current = frameIdx; }, [frameIdx]);
  useEffect(() => { framesRef.current = frames; }, [frames]);

  function mkPlayers(scene, n) {
    pmRef.current.forEach(p => scene.remove(p.g));
    pmRef.current = [];
    const curConns = connsRef.current;
    const curJoints = jointsRef.current;
    for (let i = 0; i < n; i++) {
      const c = COLORS[i % COLORS.length];
      const g = new THREE.Group();
      const bodyMat = new THREE.MeshPhongMaterial({ color: c, emissive: c, emissiveIntensity: 0.2 });
      const batMat = new THREE.MeshPhongMaterial({ color: BAT_COLOR, emissive: BAT_COLOR, emissiveIntensity: 0.35 });

      const bs = curConns.map(([a, b]) => {
        const isBat = a >= 200;
        const m = new THREE.Mesh(isBat ? batBoneGeo.clone() : bodyBoneGeo.clone(), isBat ? batMat : bodyMat);
        m.visible = false; g.add(m); return m;
      });

      const js = {};
      curJoints.forEach(id => {
        const isBat = id >= 200;
        let geo;
        if (isBat) {
          geo = id === 200 ? batKnobGeo : (id === 204 ? batBarrelGeo : batJointGeo);
        } else {
          geo = id === 83 ? headGeo : jointGeo;
        }
        const m = new THREE.Mesh(geo, isBat ? batMat : bodyMat);
        m.visible = false; js[id] = m; g.add(m);
      });

      scene.add(g);
      pmRef.current.push({ g, bs, js });
    }
  }

  const _s = new THREE.Vector3(), _e = new THREE.Vector3();
  function updMeshes(frame) {
    if (!frame?.p || !sceneRef.current) return;
    const ps = frame.p;
    const curConns = connsRef.current;
    const curJoints = jointsRef.current;
    if (pmRef.current.length !== ps.length) mkPlayers(sceneRef.current.scene, ps.length);
    ps.forEach((pl, i) => {
      const pm = pmRef.current[i]; if (!pm) return;
      pm.g.visible = pl.v !== false;
      if (!pm.g.visible) return;
      const j = pl.j || pl.bonePositions || {};
      curConns.forEach(([a, b], ci) => {
        const bone = pm.bs[ci]; if (!bone) return;
        const pa = j[a] || j[String(a)], pb = j[b] || j[String(b)];
        if (pa && pb) {
          _s.set(pa[0], pa[1], pa[2]); _e.set(pb[0], pb[1], pb[2]);
          bone.position.copy(_s); bone.lookAt(_e); bone.scale.set(1, 1, _s.distanceTo(_e)); bone.visible = true;
        } else bone.visible = false;
      });
      curJoints.forEach(id => {
        const sph = pm.js[id]; if (!sph) return;
        const p = j[id] || j[String(id)];
        if (p) { sph.position.set(p[0], p[1], p[2]); sph.visible = true; } else sph.visible = false;
      });
    });
  }

  const initScene = useCallback((el) => {
    if (sceneRef.current) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b12);
    scene.fog = new THREE.Fog(0x0b0b12, 500, 1000);
    const cam = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.1, 2000);
    const ren = new THREE.WebGLRenderer({ antialias: true });
    ren.setSize(el.clientWidth, el.clientHeight);
    ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(ren.domElement);

    scene.add(new THREE.AmbientLight(0x667788, 0.9));
    const dl = new THREE.DirectionalLight(0xffffff, 1); dl.position.set(80, 150, 40); scene.add(dl);
    scene.add(new THREE.HemisphereLight(0x88ccff, 0x224422, 0.5));

    const gnd = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), new THREE.MeshPhongMaterial({ color: 0x162816 }));
    gnd.rotation.x = -Math.PI / 2; gnd.position.set(0, -0.05, -200); scene.add(gnd);
    const drt = new THREE.Mesh(new THREE.CircleGeometry(95, 48), new THREE.MeshPhongMaterial({ color: 0x2a1a0a }));
    drt.rotation.x = -Math.PI / 2; drt.position.set(0, 0.02, -65); scene.add(drt);
    const mnd = new THREE.Mesh(new THREE.CylinderGeometry(5, 6, 0.8, 24), new THREE.MeshPhongMaterial({ color: 0x3a2510 }));
    mnd.position.set(0, 0.4, -60.5); scene.add(mnd);
    const bm = new THREE.MeshPhongMaterial({ color: 0xffffff, emissive: 0x666666 });
    [[0, 0], [63.6, -63.6], [0, -127.3], [-63.6, -63.6]].forEach(([x, z]) => { const b = new THREE.Mesh(new THREE.BoxGeometry(1.25, .3, 1.25), bm); b.position.set(x, .15, z); scene.add(b); });
    const lm = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 });
    const bp = [[0, 0], [63.6, -63.6], [0, -127.3], [-63.6, -63.6]];
    for (let i = 0; i < 4; i++) { const [x1, z1] = bp[i], [x2, z2] = bp[(i + 1) % 4]; mkC(scene, x1, .12, z1, x2, .12, z2, .12, lm); }
    const fm = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12 });
    mkC(scene, 0, .05, 0, 300, .05, -300, .1, fm); mkC(scene, 0, .05, 0, -300, .05, -300, .1, fm);
    const am = new THREE.MeshBasicMaterial({ color: 0x336633, transparent: true, opacity: 0.2 });
    for (let a = -Math.PI / 4; a < Math.PI / 4; a += .04) mkC(scene, Math.sin(a) * 330, .05, -Math.cos(a) * 330, Math.sin(a + .04) * 330, .05, -Math.cos(a + .04) * 330, .12, am);

    let o = { th: Math.PI * 0.6, ph: 0.5, r: 160, tx: 0, ty: 4, tz: -55 };
    let dr = false, pv = { x: 0, y: 0 };
    const uc = () => { cam.position.set(o.tx + o.r * Math.sin(o.ph) * Math.cos(o.th), o.ty + o.r * Math.cos(o.ph), o.tz + o.r * Math.sin(o.ph) * Math.sin(o.th)); cam.lookAt(o.tx, o.ty, o.tz); };
    uc();
    const cv = ren.domElement;
    cv.addEventListener("mousedown", e => { dr = true; pv = { x: e.clientX, y: e.clientY }; });
    cv.addEventListener("mousemove", e => { if (!dr) return; o.th -= (e.clientX - pv.x) * .006; o.ph = Math.max(.05, Math.min(1.5, o.ph - (e.clientY - pv.y) * .006)); pv = { x: e.clientX, y: e.clientY }; uc(); });
    cv.addEventListener("mouseup", () => dr = false); cv.addEventListener("mouseleave", () => dr = false);
    cv.addEventListener("wheel", e => { o.r = Math.max(5, Math.min(800, o.r + e.deltaY * .3)); uc(); }, { passive: true });

    sceneRef.current = { scene, cam, ren, setCam: (th, ph, r, tx, ty, tz) => { o.th = th; o.ph = ph; o.r = r; o.tx = tx; o.ty = ty; o.tz = tz; uc(); } };
    new ResizeObserver(() => { cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); ren.setSize(el.clientWidth, el.clientHeight); }).observe(el);

    let lt = 0, ac = 0;
    (function anim(t) {
      requestAnimationFrame(anim);
      if (playRef.current && framesRef.current?.length > 1) {
        ac += (t - lt) * speedRef.current;
        while (ac >= 33.3) { ac -= 33.3; const n = (frameRef.current + 1) % framesRef.current.length; frameRef.current = n; setFrameIdx(n); }
      }
      lt = t; if (framesRef.current?.[frameRef.current]) updMeshes(framesRef.current[frameRef.current]); ren.render(scene, cam);
    })(0);
  }, []);

  function mkC(sc, x1, y1, z1, x2, y2, z2, r, mt) {
    const s = new THREE.Vector3(x1, y1, z1), e = new THREE.Vector3(x2, y2, z2), l = s.distanceTo(e);
    const g = new THREE.CylinderGeometry(r, r, l, 4, 1); g.translate(0, l / 2, 0); g.rotateX(Math.PI / 2);
    const m = new THREE.Mesh(g, mt); m.position.copy(s); m.lookAt(e); sc.add(m);
  }

  function loadFile(file) {
    const rd = new FileReader();
    rd.onload = e => {
      try {
        const data = JSON.parse(e.target.result);
        const fc = data.connections || BODY_CONNS;
        const fj = data.jointIndices || BODY_JOINTS;
        connsRef.current = fc; jointsRef.current = fj;
        setFileConns(fc); setFileJoints(fj);
        let f = data.frames || data;
        f = f.map(fr => fr.p ? fr : { t: fr.timestamp, p: (fr.players || []).map(p => ({ v: p.visible, j: p.bonePositions || {} })) });
        setFrames(f); setFrameIdx(0); frameRef.current = 0;
        const batFrames = f.filter(fr => fr.p.some(p => p.j?.[200] || p.j?.['200'])).length;
        setInfo({ frames: f.length, players: f[0]?.p?.length || 0, bones: Object.keys(f[0]?.p?.[0]?.j || {}).length, batFrames });
        if (sceneRef.current) mkPlayers(sceneRef.current.scene, f[0]?.p?.length || 0);
      } catch (err) { alert("Error: " + err.message); }
    };
    rd.readAsText(file);
  }

  const speeds = [0.1, 0.25, 0.5, 1, 2, 4];
  const nxtSpd = () => setSpeed(s => speeds[(speeds.indexOf(s) + 1) % speeds.length]);
  const cm = (...a) => () => sceneRef.current?.setCam(...a);
  const S = a => ({ background: a ? "#0f81a" : "#ffffff08", border: `1px solid ${a ? "#0f8" : "#2a2a3a"}`, color: a ? "#0f8" : "#aaa", padding: "5px 10px", borderRadius: 3, cursor: "pointer", fontFamily: "inherit", fontSize: 11, whiteSpace: "nowrap" });

  return (
    <div style={{ width: "100%", height: "100vh", display: "flex", flexDirection: "column", background: "#0b0b12", fontFamily: "'Courier New',monospace", color: "#ccc" }}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
      onDrop={e => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); }}>
      {!frames ? (
        <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontSize: 36, marginBottom: 8 }}>⚾</div>
          <div style={{ color: "#0f8", fontSize: 20, fontWeight: "bold", letterSpacing: 2 }}>FIELDVISION REPLAY</div>
          <div style={{ color: "#555", fontSize: 12, marginBottom: 24 }}>MLB Hawk-Eye Skeletal Tracking + Bat Viewer</div>
          <label style={{ width: 360, height: 150, border: `2px dashed ${dragOver ? "#0f8" : "#0f83"}`, borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", cursor: "pointer", background: dragOver ? "#0f80a" : "transparent" }}>
            <div style={{ fontSize: 30, marginBottom: 8 }}>📂</div>
            <div style={{ fontSize: 13, color: "#888" }}>Drop capture JSON here</div>
            <div style={{ fontSize: 10, color: "#444", marginTop: 6 }}>fieldvision_with_bat.json or fieldvision_correct.json</div>
            <input type="file" accept=".json" style={{ display: "none" }} onChange={e => e.target.files[0] && loadFile(e.target.files[0])} />
          </label>
        </div>
      ) : (
        <>
          <div ref={el => { if (el && !sceneRef.current) { initScene(el); if (frames) mkPlayers(sceneRef.current.scene, frames[0]?.p?.length || 0); } }} style={{ flex: 1, position: "relative" }}>
            <div style={{ position: "absolute", top: 10, left: 14, color: "#0f8", fontSize: 13, fontWeight: "bold", letterSpacing: 1, pointerEvents: "none", textShadow: "0 0 8px #0f84" }}>⚾ FIELDVISION REPLAY</div>
            {info && (
              <div style={{ position: "absolute", top: 12, right: 14, fontSize: 11, color: "#555", pointerEvents: "none" }}>
                <span style={{ color: "#0f8" }}>{info.players}</span> players · <span style={{ color: "#0f8" }}>{info.bones}</span> joints · <span style={{ color: "#0f8" }}>{info.frames}</span> frames
                {info.batFrames > 0 && <> · <span style={{ color: "#fd4" }}>{info.batFrames} bat frames</span></>}
              </div>
            )}
          </div>
          <div style={{ padding: "8px 14px 10px", background: "#0c0c14", borderTop: "1px solid #1a1a2a" }}>
            <input type="range" min={0} max={frames.length - 1} value={frameIdx} onChange={e => { const v = +e.target.value; setFrameIdx(v); frameRef.current = v; }} style={{ width: "100%", accentColor: "#0f8", height: 3, marginBottom: 6 }} />
            <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
              <button onClick={() => setPlaying(!playing)} style={S(playing)}>{playing ? "⏸ Pause" : "▶ Play"}</button>
              <button onClick={nxtSpd} style={S(false)}>{speed}x</button>
              <div style={{ width: 1, height: 16, background: "#222" }} />
              <button onClick={cm(Math.PI, .38, 80, 0, 3, -25)} style={S(false)}>Behind HP</button>
              <button onClick={cm(0, .1, 500, 0, 0, -140)} style={S(false)}>Bird's Eye</button>
              <button onClick={cm(Math.PI / 2, .32, 200, 0, 3, -80)} style={S(false)}>1B Line</button>
              <button onClick={cm(.02, .55, 40, 0, 4, -58)} style={S(false)}>Mound</button>
              <button onClick={cm(Math.PI * .6, .5, 160, 0, 4, -55)} style={S(false)}>Default</button>
              <label style={{ ...S(false), cursor: "pointer" }}>📂 Load<input type="file" accept=".json" style={{ display: "none" }} onChange={e => e.target.files[0] && loadFile(e.target.files[0])} /></label>
              <div style={{ marginLeft: "auto", fontSize: 11, color: "#555" }}><span style={{ color: "#0f8" }}>{frameIdx}</span>/{frames.length - 1}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
