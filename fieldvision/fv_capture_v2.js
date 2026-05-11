/**
 * MLB FieldVision Complete Capture Tool v2
 * =========================================
 * Captures skeletal tracking + bat + player MLB IDs
 * 
 * Usage:
 *   1. Open any LIVE game in Gameday 3D view on mlb.com
 *   2. Make sure the 3D view is active (click "3D" button if needed)
 *   3. Open DevTools console (F12 → Console)  
 *   4. Paste this entire script and press Enter
 *   5. Wait for "READY" message
 *   6. FVCapture.start(10)   — capture 10 seconds
 *   7. FVCapture.download()  — save JSON with player IDs + skeleton + bat
 */
(function() {
  'use strict';
  
  // ═══════════════════════════════════════════════
  // CORRECT skeleton.bones[] array indices
  // (NOT boneIdMap IDs — those are different!)
  // ═══════════════════════════════════════════════
  const SKEL_JOINTS = [0,1,2,3,6,9,14,42,45,48,53,81,83,86,87,90,91,95,98,99];
  const SKEL_CONNS = [
    [0,1],[1,2],[2,81],[81,83],      // Spine: Pelvis→TorsoA→TorsoB→Neck→Head
    [0,86],[86,95],[95,98],[98,99],   // R Leg: Pelvis→HipMaster→HipRT→KneeRT→FootRT
    [86,87],[87,90],[90,91],          // L Leg: HipMaster→HipLT→KneeLT→FootLT
    [2,42],[42,45],[45,48],[48,53],   // R Arm: TorsoB→ClavRT→ShouldRT→ElbowRT→HandRT
    [2,3],[3,6],[6,9],[9,14]          // L Arm: TorsoB→ClavLT→ShouldLT→ElbowLT→HandLT
  ];
  const BAT_CONNS = [[200,201],[201,202],[202,203],[203,204]];
  
  const JOINT_NAMES = {
    0:'Pelvis',1:'TorsoA',2:'TorsoB',3:'ClavicleLT',6:'ShoulderLT',
    9:'ElbowLT',14:'HandLT',42:'ClavicleRT',45:'ShoulderRT',
    48:'ElbowRT',53:'HandRT',81:'Neck',83:'Head',86:'HipMaster',
    87:'HipLT',90:'KneeLT',91:'FootLT',95:'HipRT',98:'KneeRT',99:'FootRT',
    200:'BatBottom',201:'BatHandle',202:'BatBody',203:'BatSpot',204:'BatTop'
  };
  
  // ═══════════════════════════════════════════════
  // Find poserContext
  // ═══════════════════════════════════════════════
  function findPoserContext() {
    const fvDiv = document.querySelector('[class*="FieldVisionPlayerContainer"]') 
                || document.querySelector('[class*="FieldVisionApp"]');
    if (!fvDiv) return null;
    const fk = Object.keys(fvDiv).find(k => k.startsWith('__reactFiber'));
    if (!fk) return null;
    let fiber = fvDiv[fk];
    for (let d = 0; d < 30 && fiber; d++) {
      if (fiber.memoizedState) {
        let hook = fiber.memoizedState;
        for (let i = 0; i < 25 && hook; i++) {
          try {
            if (hook.memoizedState?.current?.poserContext)
              return hook.memoizedState.current.poserContext;
          } catch(e) {}
          hook = hook.next;
        }
      }
      fiber = fiber.return;
    }
    return null;
  }
  
  // ═══════════════════════════════════════════════
  // Extract player ID mapping
  // ═══════════════════════════════════════════════
  function buildPlayerMap(pc) {
    const map = new Map(); // armature Object3D → { mlbId, type, name }
    
    // Method 1: world.actors.actors Map
    try {
      const actorsObj = pc.world?.actors;
      if (actorsObj?.actors) {
        const inner = actorsObj.actors;
        if (inner.forEach) {
          inner.forEach((actor, key) => {
            // Find the Armature inside this actor entity
            let armature = null;
            if (actor.traverse) {
              actor.traverse(child => {
                if (!armature && child.name === 'Armature') armature = child;
              });
            }
            // Try to find the MLB ID
            const id = actor.actorId || actor.id || actor.mlbId || actor.playerId || key;
            const type = actor.actorType || actor.type || 'unknown';
            if (armature) map.set(armature, { mlbId: id, type });
          });
        }
      }
    } catch(e) { console.warn('[FVCapture] actors method failed:', e.message); }
    
    // Method 2: Fall back to bufferManager.labels for current segment
    if (map.size === 0) {
      try {
        const labels = pc.bufferManager?.labels;
        if (labels) {
          const keys = Object.keys(labels);
          const latest = labels[keys[keys.length - 1]];
          if (latest) {
            console.log('[FVCapture] Using labels fallback. Latest segment actor:', latest.actor, 'type:', latest.type);
          }
        }
      } catch(e) {}
    }
    
    // Method 3: Get boxscore for name lookups
    try {
      const boxscore = pc.gameState?.boxscore;
      if (boxscore?.teams) {
        window.__boxscore = boxscore;
        console.log('[FVCapture] Boxscore available for player name lookups');
      }
    } catch(e) {}
    
    return map;
  }
  
  // ═══════════════════════════════════════════════
  // Pre-cache skeleton references
  // ═══════════════════════════════════════════════
  function cacheSkeletons(pc, playerMap) {
    const world = pc.world;
    const skels = [];
    const armSet = new Set();
    let batMesh = null;
    
    world.traverse(obj => {
      if (obj.name === 'Bat' && obj.skeleton?.bones?.length === 5) batMesh = obj;
      if (obj.skeleton?.bones?.length === 103) {
        const arm = obj.parent;
        if (armSet.has(arm)) return;
        armSet.add(arm);
        
        const playerInfo = playerMap.get(arm) || {};
        skels.push({ arm, bones: obj.skeleton.bones, ...playerInfo });
      }
    });
    
    return { skels, batMesh };
  }
  
  // ═══════════════════════════════════════════════
  // Main capture object
  // ═══════════════════════════════════════════════
  window.FVCapture = {
    frames: [],
    playerMap: null,
    cache: null,
    pc: null,
    intervalId: null,
    
    init() {
      this.pc = findPoserContext();
      if (!this.pc) {
        console.error('[FVCapture] No poserContext found. Make sure 3D view is active!');
        return false;
      }
      this.playerMap = buildPlayerMap(this.pc);
      this.cache = cacheSkeletons(this.pc, this.playerMap);
      
      console.log(`[FVCapture] READY!`);
      console.log(`  Players: ${this.cache.skels.length}`);
      console.log(`  Bat: ${this.cache.batMesh ? 'found' : 'not found'}`);
      console.log(`  Player IDs mapped: ${this.playerMap.size}`);
      
      // Log player roster
      this.cache.skels.forEach((s, i) => {
        console.log(`  #${i}: mlbId=${s.mlbId || '?'} type=${s.type || '?'}`);
      });
      
      return true;
    },
    
    start(seconds = 10) {
      if (!this.pc && !this.init()) return;
      this.frames = [];
      const maxFrames = seconds * 20; // ~20fps capture rate
      let count = 0;
      
      console.log(`[FVCapture] Capturing ${seconds}s (${maxFrames} frames)...`);
      
      this.intervalId = setInterval(() => {
        if (count >= maxFrames) {
          clearInterval(this.intervalId);
          console.log(`[FVCapture] Done! ${this.frames.length} frames captured.`);
          console.log(`[FVCapture] Run FVCapture.download() to save.`);
          return;
        }
        
        const frame = { t: Date.now(), p: [] };
        
        for (const skel of this.cache.skels) {
          const j = {};
          
          // Player joints
          for (const idx of SKEL_JOINTS) {
            const e = skel.bones[idx]?.matrixWorld?.elements;
            if (e) j[idx] = [+(e[12].toFixed(2)), +(e[13].toFixed(2)), +(e[14].toFixed(2))];
          }
          
          // Bat (check if this armature owns it)
          let hasBat = false;
          if (this.cache.batMesh) {
            const batParent = this.cache.batMesh.parent;
            if (batParent === skel.arm) {
              hasBat = true;
              this.cache.batMesh.skeleton.bones.forEach((bb, bi) => {
                const e = bb?.matrixWorld?.elements;
                if (e) j[200+bi] = [+(e[12].toFixed(2)), +(e[13].toFixed(2)), +(e[14].toFixed(2))];
              });
            }
          }
          
          // Also proximity-match bat if not direct child
          if (!hasBat && this.cache.batMesh && j[53]) { // HandRT exists
            const hand = j[53];
            const batHandle = this.cache.batMesh.skeleton.bones[1];
            if (batHandle?.matrixWorld) {
              const he = batHandle.matrixWorld.elements;
              const dist = Math.sqrt((hand[0]-he[12])**2 + (hand[1]-he[13])**2 + (hand[2]-he[14])**2);
              if (dist < 2) {
                hasBat = true;
                this.cache.batMesh.skeleton.bones.forEach((bb, bi) => {
                  const e = bb?.matrixWorld?.elements;
                  if (e) j[200+bi] = [+(e[12].toFixed(2)), +(e[13].toFixed(2)), +(e[14].toFixed(2))];
                });
              }
            }
          }
          
          frame.p.push({
            v: skel.arm.visible,
            j,
            bat: hasBat,
            mlbId: skel.mlbId || null,
            type: skel.type || null
          });
        }
        
        this.frames.push(frame);
        count++;
        if (count % 20 === 0) console.log(`[FVCapture] ${count}/${maxFrames} frames`);
      }, 50);
    },
    
    stop() {
      if (this.intervalId) clearInterval(this.intervalId);
      console.log(`[FVCapture] Stopped. ${this.frames.length} frames.`);
    },
    
    download() {
      const data = {
        captureTime: new Date().toISOString(),
        game: document.title,
        connections: [...SKEL_CONNS, ...BAT_CONNS],
        jointIndices: [...SKEL_JOINTS, 200, 201, 202, 203, 204],
        jointNames: JOINT_NAMES,
        players: this.cache.skels.map((s, i) => ({
          index: i,
          mlbId: s.mlbId || null,
          type: s.type || null
        })),
        frameCount: this.frames.length,
        frames: this.frames
      };
      
      const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'fv_capture_' + Date.now() + '.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      console.log(`[FVCapture] Downloaded ${data.frameCount} frames with ${data.players.length} players`);
    },
    
    status() {
      console.log(`
═══ FVCapture Status ═══
Frames:     ${this.frames.length}
Players:    ${this.cache?.skels?.length || 0}
IDs mapped: ${this.playerMap?.size || 0}
Bat:        ${this.cache?.batMesh ? 'yes' : 'no'}
      `);
    }
  };
  
  // Auto-init
  if (FVCapture.init()) {
    console.log(`
╔══════════════════════════════════════════════════╗
║  FVCapture v2 — Ready!                           ║
║                                                  ║
║  FVCapture.start(10)    — capture 10 seconds     ║
║  FVCapture.download()   — save JSON              ║
║  FVCapture.status()     — check progress         ║
║  FVCapture.stop()       — stop early             ║
╚══════════════════════════════════════════════════╝
    `);
  }
})();
